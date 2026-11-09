#!/usr/bin/env python3
"""Fetch a role bundle from a short-lived presigned R2 URL, verify sha256, extract.

usage: bundle_fetch.py ROLE BYTES SHA256 DEST
stdin JSON: {"url": "<presigned GET>"}  (deleted from the caller's stdin; never logged)
stdout last line JSON: {sha_ok, tar_rc, seconds, gbps, bytes, role, dest}

Port of the host RA helper used by run5090 / drive_item.bundle_trainer
(E1041–E1048 on real B300s). Extract into one filesystem (e.g. /opt/plain/tbx)
to avoid EXDEV on Lium /root loop mounts.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 4:
        sys.stderr.write('usage: bundle_fetch.py ROLE BYTES SHA256 DEST\n')
        return 2
    role, nbytes_s, expect, dest = argv
    try:
        nbytes = int(nbytes_s)
    except ValueError:
        sys.stderr.write('BYTES must be int\n')
        return 2
    if len(expect) != 64:
        sys.stderr.write('SHA256 must be hex64\n')
        return 2
    spec = json.load(sys.stdin)
    url = spec.get('url') if isinstance(spec, dict) else None
    if not isinstance(url, str) or not url.startswith('http'):
        sys.stderr.write('stdin must be JSON {"url": "https://..."}\n')
        return 2

    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    h = hashlib.sha256()
    got = 0
    tar_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix='bun-%s-' % role, suffix='.tar', delete=False) as tmp:
            tar_path = tmp.name
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=3600) as resp:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
        digest = h.hexdigest()
        sha_ok = digest == expect and (nbytes <= 0 or got == nbytes)
        tar_rc = 1
        if sha_ok:
            p = subprocess.run(
                ['tar', '--no-same-owner', '-xf', tar_path, '-C', dest],
                capture_output=True, timeout=3600,
            )
            tar_rc = int(p.returncode)
        seconds = round(time.time() - t0, 1)
        gbps = round(got * 8 / seconds / 1e9, 2) if seconds > 0 else None
        print(json.dumps(dict(
            role=role, dest=dest, bytes=got, expect_bytes=nbytes,
            sha256=digest, sha_ok=bool(sha_ok), tar_rc=tar_rc,
            seconds=seconds, gbps=gbps,
        )))
        return 0 if sha_ok and tar_rc == 0 else 1
    finally:
        if tar_path:
            try:
                os.remove(tar_path)
            except OSError:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
