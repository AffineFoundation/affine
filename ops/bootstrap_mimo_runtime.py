#!/usr/bin/env python3
"""Install a pinned MiMo worker into a NEW directory; never activate a service.

The lock file must be the captured, fully pinned runtime used for qualification.
Use the exact TOPLOC wheel rebuilt against torch 2.14/cu130, not PyPI's older
prebuilt binary. Native environment/grader installation is separately pinned by
the epoch source contract. No model, credentials, service or fleet is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'
REQUIRED = {'torch': ('2.14.0', '2.14.0+cu130'), 'transformers': ('5.14.1',),
            'toploc': ('0.1.6',), 'safetensors': ('0.8.0',),
            'tokenizers': ('0.22.2',), 'numpy': ('2.5.3',)}


def checked_bytes(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('ordinary pinned input file required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('pinned input hash mismatch: ' + path.name)
    return raw


def resolved_lock(raw):
    """No dependency resolution, moving URLs, shell flags or unpinned extras."""
    result = {}
    for row in raw.decode().splitlines():
        row = row.strip()
        if not row or row.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)', row)
        if not match:
            raise ValueError('requirements must contain only exact package==version pins')
        name = re.sub(r'[-_.]+', '-', match[1]).lower()
        if name in result:
            raise ValueError('duplicate normalized package pin')
        result[name] = match[2]
    for name, versions in REQUIRED.items():
        if result.get(name) not in versions:
            raise ValueError('qualification runtime pin differs: ' + name)
    return result


def commands(destination, uv, python, wheel, *, torch_pin):
    env = Path(destination) / 'venv'
    interpreter = str(env / 'bin/python')
    if torch_pin not in REQUIRED['torch']:
        raise ValueError('exact captured torch package pin required')
    # PyPI's 2.14.0 distribution itself contains the cu130 runtime. Replacing it
    # with the separately indexed 2.14.0+cu130 package changes the signed package
    # fingerprint, even when the loaded torch.__version__ and source are equal.
    torch_index = ['--index-url', TORCH_INDEX] if torch_pin.endswith('+cu130') else []
    return [
        [uv, 'venv', '--python', python, str(env)],
        [uv, 'pip', 'install', '--python', interpreter, '--no-deps',
         *torch_index, 'torch==' + torch_pin],
        [uv, 'pip', 'install', '--python', interpreter, '--no-deps',
         '--requirements', str(Path(destination) / 'remaining-requirements.txt')],
        [uv, 'pip', 'install', '--python', interpreter, '--no-deps', str(wheel)],
        [uv, 'pip', 'check', '--python', interpreter],
    ]


PROBE = r'''
import json, sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from subnet.mimo_runtime import runtime_fingerprint
from subnet.proofs import verify_mapped_proofs
from toploc import build_proofs_base64
torch.set_num_threads(2)
fingerprint = runtime_fingerprint()
acts = [torch.linspace(-3, 3, 128).reshape(2, 64).to(torch.bfloat16)]
proofs = build_proofs_base64(acts, decode_batching_size=16, topk=16)
results = verify_mapped_proofs(acts, proofs, decode_batching_size=16, topk=16)
if not results or any(r.exp_mismatches or r.mant_err_mean for r in results):
    raise ValueError('TOPLOC native build/verification roundtrip failed')
if not torch.cuda.is_available():
    raise ValueError('GPU worker requires available CUDA')
report = dict(version='mimo-runtime-bootstrap-v1', qualified=False,
    fingerprint=fingerprint, torch_runtime=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(), toploc_native_roundtrip=True,
    model_downloaded=False, services_activated=False)
Path(sys.argv[2]).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
'''


def bootstrap(args, *, runner=subprocess.run):
    destination = Path(args.destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('destination must be new; existing workers are never modified')
    if destination.parent.resolve() != destination.parent or not destination.parent.is_dir():
        raise ValueError('ordinary existing parent directory required')
    if not 1 <= args.max_jobs <= 4:
        raise ValueError('MAX_JOBS must be between one and four')
    lock_raw = checked_bytes(args.requirements, args.requirements_sha256)
    pins = resolved_lock(lock_raw)
    wheel = Path(args.toploc_wheel).absolute()
    checked_bytes(wheel, args.toploc_wheel_sha256)
    if not re.fullmatch(r'toploc-0\.1\.6-cp312-cp312-(?:linux|manylinux[^-]*)_x86_64\.whl', wheel.name):
        raise ValueError('the qualified CPython3.12 x86_64 TOPLOC wheel is required')
    source = Path(args.source_root).absolute()
    if not (source / 'subnet/mimo_runtime.py').is_file():
        raise ValueError('staged MiMo source required for import/ABI probe')
    plan = commands(destination, args.uv, args.python, wheel, torch_pin=pins['torch'])
    if args.plan:
        return dict(commands=plan, destination=str(destination), qualified=False,
                    model_downloaded=False, services_activated=False)
    destination.mkdir(mode=0o700)
    (destination / 'requirements.original.txt').write_bytes(lock_raw)
    (destination / 'remaining-requirements.txt').write_text(''.join(
        name + '==' + version + '\n' for name, version in sorted(pins.items())
        if name not in ('torch', 'toploc')))
    record = dict(version='mimo-runtime-bootstrap-attempt-v1', started_at=time.time(),
        requirements_sha256=args.requirements_sha256, toploc_wheel_sha256=args.toploc_wheel_sha256,
        completed=False, source_root=str(source), qualified=False)
    env = dict(os.environ, MAX_JOBS=str(args.max_jobs), OMP_NUM_THREADS=str(args.max_jobs),
               UV_CACHE_DIR=str(destination / 'package-cache'))
    try:
        with (destination / 'install.log').open('xb') as log:
            for command in plan:
                runner(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            probe = [str(destination / 'venv/bin/python'), '-B', '-c', PROBE,
                     str(source), str(destination / 'runtime-probe.json')]
            runner(probe, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['completed'] = True
        # The new venv owns installed files. Retaining another wheel/download
        # cache here would consume trainer space without helping normal epochs.
        shutil.rmtree(destination / 'package-cache', ignore_errors=True)
        return record
    finally:
        record['finished_at'] = time.time()
        (destination / 'BOOTSTRAP.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--requirements', required=True)
    parser.add_argument('--requirements-sha256', required=True)
    parser.add_argument('--toploc-wheel', required=True)
    parser.add_argument('--toploc-wheel-sha256', required=True)
    parser.add_argument('--source-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--python', default='3.12')
    parser.add_argument('--uv', default='uv')
    parser.add_argument('--max-jobs', type=int, default=2)
    parser.add_argument('--plan', action='store_true')
    print(json.dumps(bootstrap(parser.parse_args()), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
