#!/usr/bin/env python3
"""Build / verify the UNSIGNED host-v834f admission proposal for Jacob ROOT signing.

Default is --dry-run: never reads the ROOT seed. Real signing must use Jacob's
existing ROOT process (lifecycle authority_seed_file) — same pattern as
ops/ff_train_params/sign_v833j.py. Do not bootstrap or override ROOT.

Usage:
  python3 sign_host_v834f.py --dry-run
  python3 sign_host_v834f.py --base-private /path/to/host-v834e/TRAINER-HOST...private.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'stage'
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'


def need(ok: bool, why: str) -> None:
    if not ok:
        raise SystemExit('REFUSED: ' + why)


def hf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', default=True,
                    help='default: never touch ROOT seed (safe)')
    ap.add_argument('--sign', action='store_true',
                    help='operator-only: perform ROOT sign (requires --base-private + seed)')
    ap.add_argument('--base-private', type=Path, default=None,
                    help='prior ROOT-admitted TRAINER-HOST…private.json (host-v834e)')
    args = ap.parse_args(list(argv) if argv is not None else None)
    dry = not args.sign

    need(STAGE.is_dir(), 'stage/ missing — regenerate host_v834f_signing artifact')
    proposal = json.loads((HERE / 'PROPOSAL.unsigned.json').read_text())
    host_files = proposal['host_files']
    for rel, expect in host_files.items():
        path = STAGE / rel
        need(path.is_file(), 'host file missing (fail closed): %s' % rel)
        got = hf(path)
        need(got == expect, 'digest mismatch %s' % rel)

    # Manifest: every listed path must exist; no .pyc entries.
    man = (HERE / 'SIGNED-MANIFEST.sha256').read_text().splitlines()
    for line in man:
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition('  ')
        need(rel.startswith('./trainer-host-v834f/'), 'bad manifest path: %s' % rel)
        need(not rel.endswith('.pyc') and '/__pycache__/' not in rel, 'pyc in manifest: %s' % rel)
        local = STAGE / rel[len('./trainer-host-v834f/'):]
        need(local.is_file(), 'manifest file missing (fail closed): %s' % rel)
        need(hf(local) == digest, 'manifest digest mismatch: %s' % rel)

    out = {
        'ok': True,
        'dry_run': dry,
        'host_version': 'host-v834f',
        'host_file_count': len(host_files),
        'manifest_sha256': proposal['manifest_sha256'],
        'todo': str(HERE / 'TODO-ROOT-SIGN.md'),
        'root_id': ROOT_ID,
    }
    if dry:
        out['next'] = (
            'Hand PROPOSAL.unsigned.json + stage/ + SIGNED-MANIFEST.sha256 to Jacob. '
            'Re-run with --sign --base-private <v834e ROOT-ADMITTED> on the operator host.'
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    need(args.base_private is not None and args.base_private.is_file(),
         '--base-private required for --sign')
    # Real sign path intentionally minimal here: operator should prefer the
    # established sign_v833j-style script on the ROOT host. Refuse silent
    # seed discovery from this checkout.
    need(False, 'use Jacob ROOT host sign flow (see TODO-ROOT-SIGN.md); refusing in-repo seed read')


if __name__ == '__main__':
    main()
