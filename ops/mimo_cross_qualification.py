#!/usr/bin/env python3
"""Audit original diagnostic attempts on another device without altering them.

The input manifest and miner identity are operator-authenticated arguments. This
does not sign a production calibration or extrapolate an error bound from a few
rollouts. A rejection is recorded for investigation, not a miner penalty.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.storage import canonical, sha
from subnet.complete_group_codec import decode, limits
from subnet.complete_group_runtime import MimoGroupRuntime
from subnet.complete_group_verification import verify_rollout
from subnet.audit_policy import InvalidSample


def run(args):
    import torch
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(args.manifest).read_bytes()
    if sha(raw) != args.manifest_sha256:
        raise ValueError('original operator-pinned diagnostic manifest required')
    manifest = json.loads(raw)
    if not manifest['epoch'].startswith('private-'):
        raise ValueError('private diagnostic only')
    if len(args.attempt) > manifest['group_contract']['group_size']:
        raise ValueError('bounded original attempt count')
    definition = manifest['environments'][0]
    backend = None
    began = time.monotonic()
    report = dict(version='mimo-cross-device-diagnostic-v1', qualification=False,
                  manifest_sha256=args.manifest_sha256, miner=args.miner,
                  checkpoint=manifest['checkpoint']['id'],
                  gpu=torch.cuda.get_device_name(), attempts=[])
    try:
        backend = MimoGroupRuntime(args.checkpoint, manifest, definition['spec'],
                                   definition['harness'], miner=args.miner)
        report['load_seconds'] = time.monotonic() - began
        for name in args.attempt:
            path = Path(name)
            data = path.read_bytes()
            metadata, tensors = decode(data, limits(manifest))[0]
            if set(metadata) != {'original_rollout'} or len(tensors) != 1:
                raise ValueError('original recorded single-attempt artifact required')
            row = metadata['original_rollout']
            at = time.monotonic()
            result = dict(nonce=row['seed'], index=row['index'],
                          artifact_sha256=sha(data), artifact_bytes=len(data),
                          tokens=len(row['turns'][0]['output']))
            try:
                result.update(status='accepted', verification=verify_rollout(backend, row, tensors[0]))
            except InvalidSample as error:
                result.update(status='rejected-requires-investigation', error=str(error))
            result['seconds'] = time.monotonic() - at
            report['attempts'].append(result)
            print(json.dumps(result), flush=True)
        report['status'] = ('all-observed-attempts-accepted' if all(
            r['status'] == 'accepted' for r in report['attempts']) else 'observed-disagreement')
    except BaseException as error:
        report.update(status='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic() - began
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(canonical(report))
        if backend is not None:
            backend.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'manifest-sha256', 'miner', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--attempt', action='append', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
