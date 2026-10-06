"""Credentialless, explicit small-object R2 probe; never a training benchmark.

An operator supplies short-lived, exact-object PUT/GET capabilities. Payloads
live only in bounded RAM; durable probe objects and the report remain in R2.
Neither model files nor optimizer state are read or changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def run(plan):
    import requests
    if plan.get('version') != 'bounded-random-transport-probe-v1':
        raise ValueError('explicit transport-only plan')
    size = plan['object_bytes']
    groups = plan['groups']
    if type(size) is not int or not 1 <= size <= 64 * 1024**2:
        raise ValueError('bounded object size')
    if type(groups) is not list or not 1 <= len(groups) <= 3:
        raise ValueError('bounded profile count')
    total = 0
    for group in groups:
        if type(group['concurrency']) is not int or group['concurrency'] not in (1, 4, 8):
            raise ValueError('bounded concurrency')
        if not 1 <= len(group['objects']) <= 8:
            raise ValueError('bounded object count')
        total += size * len(group['objects'])
        for obj in group['objects']:
            for kind in ('put_url', 'get_url'):
                url = urlsplit(obj[kind])
                if url.scheme != 'https' or url.hostname != plan['endpoint_host'] or url.username or url.password:
                    raise ValueError('exact HTTPS storage endpoint')
    if total > 1024**3:
        raise ValueError('bounded total probe traffic')

    def transfer(obj):
        if time.time() >= plan['expires_at']:
            raise TimeoutError('probe capability expired')
        payload = os.urandom(size)
        expected = hashlib.sha256(payload).hexdigest()
        started = time.monotonic()
        response = requests.put(obj['put_url'], data=payload, timeout=(10, 120), allow_redirects=False)
        if response.status_code not in (200, 201, 204):
            raise ValueError('probe PUT refused')
        response.close()
        uploaded = time.monotonic()
        del payload
        sha = hashlib.sha256()
        count = 0
        with requests.get(obj['get_url'], stream=True, timeout=(10, 120), allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError('probe GET refused')
            for block in response.iter_content(1024**2):
                count += len(block)
                if count > size:
                    raise ValueError('probe readback exceeded bound')
                sha.update(block)
        if count != size or sha.hexdigest() != expected:
            raise ValueError('full probe readback mismatch')
        return dict(key=obj['key'], bytes=size, sha256=expected,
                    PUT_seconds=uploaded-started, GET_seconds=time.monotonic()-uploaded,
                    full_readback_verified=True)

    results = []
    for group in groups:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=group['concurrency']) as pool:
            rows = list(pool.map(transfer, group['objects']))
        results.append(dict(concurrency=group['concurrency'], objects=rows,
                            PUT_and_GET_wall_seconds=time.monotonic()-started))
    return dict(version=plan['version'], groups=results, model_or_optimizer_touched=False,
                local_payload_files_created=False, full_state_throughput_inferred=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    args = parser.parse_args()
    try:
        print(json.dumps(run(json.loads(Path(args.plan).read_bytes()))))
    except Exception as error:
        # Presigned capability URLs must never appear in human-facing errors.
        print(json.dumps(dict(failed=True, error_type=type(error).__name__)))
        raise SystemExit(1)
