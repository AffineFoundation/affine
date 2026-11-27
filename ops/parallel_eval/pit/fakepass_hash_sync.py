#!/usr/bin/env python3
"""Sandbox stand-in for sync_expected_hash.py, wired via PE_SYNC_EXPECTED_HASH.

Fakepass copies this file into ``<sandbox>/hash-sync/`` and pe_apply's
record_applied hook runs it with ``--head=<C>``. It applies the same lookup
rules the next box depends on (promoted index has weight shards, meta.json and
a ``{"r2_key"}`` meta pointer), then writes the expected-master hash file next
to itself. It never reaches the live trainer pod (no rssh) and refuses to run
outside a fakepass sandbox.

Exit codes: 0 updated / no-op, 2 lookup failed (ALERT written, hash unchanged).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

HEX64 = re.compile(r'^[0-9a-f]{64}$')
STATE = Path(__file__).resolve().parent
HASH_FILE = STATE / 'fp32-expected-master.hash'
ALERT = STATE / 'ALERT.json'
LAST = STATE / 'LAST-SYNC.json'


def _in_sandbox(path: Path) -> bool:
    tmp = Path(tempfile.gettempdir()).resolve()
    return str(path.resolve()).startswith(str(tmp) + os.sep) and 'pit-fakepass-' in str(path)


def lookup(head: str) -> dict:
    idx_root = os.environ.get('PIT_FP32_MASTER_INDEX') or ''
    if not idx_root or not _in_sandbox(Path(idx_root)):
        return {'ok': False, 'alert': 'PIT_FP32_MASTER_INDEX not a fakepass sandbox dir'}
    path = Path(idx_root) / ('%s.json' % head)
    if not path.is_file():
        return {'ok': False, 'alert': 'index-missing', 'index': str(path)}
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        return {'ok': False, 'alert': 'index-unreadable:%s' % str(exc)[:80]}
    digest = str(doc.get('master_digest') or '')
    files = doc.get('files') or {}
    shards = [n for n in files if str(n).endswith('.safetensors')]
    meta_ptr = doc.get('meta_json')
    problems = []
    if not HEX64.match(digest):
        problems.append('master_digest')
    if not HEX64.match(str(doc.get('bf16_sha256') or '')):
        problems.append('bf16_sha256')
    if not shards:
        problems.append('no-weight-shards')
    if 'meta.json' not in files:
        problems.append('no-meta.json')
    if not isinstance(meta_ptr, dict) or set(meta_ptr) != {'r2_key'}:
        problems.append('meta_json-not-r2_key-pointer')
    elif (files.get('meta.json') or {}).get('r2_key') != meta_ptr['r2_key']:
        problems.append('meta_json-pointer-mismatch')
    if doc.get('pending') is not False:
        problems.append('index-not-promoted')
    if problems:
        return {'ok': False, 'alert': 'index-incomplete:' + ','.join(problems), 'index': str(path)}
    return {
        'ok': True,
        'master_digest': digest,
        'bf16_sha256': doc['bf16_sha256'],
        'shards': len(shards),
        'index': str(path),
    }


def main(argv: list) -> int:
    if not _in_sandbox(STATE):
        print(json.dumps({'ok': False, 'error': 'refuse: not inside a fakepass sandbox'}))
        return 2
    head = None
    for a in argv[1:]:
        if a.startswith('--head='):
            head = a.split('=', 1)[1]
        elif HEX64.match(a):
            head = a
    if not head or not HEX64.match(head):
        print(json.dumps({'ok': False, 'error': '--head=<hex64> required'}))
        return 2
    res = lookup(head)
    prev = HASH_FILE.read_text().strip() if HASH_FILE.is_file() else None
    out = {'at': time.time(), 'head': head, 'lookup': res, 'previous_hash': prev, 'action': None}
    if not res.get('ok'):
        out['action'] = 'alert-fail-closed-keep-previous'
        ALERT.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(out))
        return 2
    digest = res['master_digest']
    out['action'] = 'noop-already-current' if prev == digest else 'updated'
    if prev != digest:
        tmp = HASH_FILE.with_suffix('.tmp')
        tmp.write_text(digest + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, HASH_FILE)
        out['new_hash'] = digest
    LAST.write_text(json.dumps(out, indent=2) + '\n')
    if ALERT.exists():
        ALERT.unlink()
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
