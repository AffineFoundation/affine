#!/usr/bin/env python3
"""One-time ROOT ceremony: sign a PIT host admission key.

Operator runs this **on the host** (or an air-gapped ROOT machine) after
generating a host key. Never embeds or commits key material.

Usage (operator):

  # 1) On the validator host, generate a host key (0600) — do this once:
  python3 - <<'PY'
  from nacl.signing import SigningKey
  from pathlib import Path
  sk = SigningKey.generate()
  Path('/secure/pit-host.key').write_bytes(bytes(sk))  # or .hex()
  Path('/secure/pit-host.key').chmod(0o600)
  print(bytes(sk.verify_key).hex())
  PY

  # 2) On the ROOT machine (with ROOT seed available out-of-band only):
  AFFINE_PIT_ROOT_SEED_FILE=/secure/root.seed \\
    python3 -m ops.parallel_eval.pit.ceremony_sign_host_key \\
      --host-pubkey-hex <hex from step 1> \\
      --out /secure/pit-host.cert.json \\
      --ttl-days 365

  # 3) Install on the validator (cert is public; host key stays 0600):
  #   AFFINE_PIT_HOST_KEY_FILE=/secure/pit-host.key
  #   AFFINE_PIT_HOST_CERT_FILE=/secure/pit-host.cert.json
  #   AFFINE_PIT_ROOT_PUBKEY_HEX=<root verify key hex>

Refuses to run if --out would overwrite without --force. Never prints seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .signer import PLACEHOLDER_ROOT_PUBKEY_HEX, sign_host_key_cert


def _load_root_signing_key(seed_file: str):
    from nacl.signing import SigningKey
    p = Path(seed_file)
    if not p.is_file():
        raise SystemExit('ROOT seed file not found')
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise SystemExit('ROOT seed file must be mode 0600 (or tighter)')
    raw = p.read_bytes()
    # Hex text may have a trailing newline; raw 32-byte seeds must not be stripped
    # (a seed byte can legitimately be 0x0a / 0x20).
    text = raw.strip()
    if len(text) == 64 and all(chr(c) in '0123456789abcdefABCDEF' for c in text):
        seed = bytes.fromhex(text.decode('ascii'))
    elif len(raw) == 32:
        seed = raw
    else:
        raise SystemExit('ROOT seed file must be 32 raw bytes or 64 hex chars')
    return SigningKey(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='ROOT ceremony: sign PIT host admission key')
    ap.add_argument('--host-pubkey-hex', required=True, help='32-byte ed25519 verify key as 64 hex chars')
    ap.add_argument('--out', required=True, help='path to write host key cert JSON')
    ap.add_argument('--ttl-days', type=float, default=365.0)
    ap.add_argument('--host-id', default='affinevalidator')
    ap.add_argument('--root-seed-file', default=os.environ.get('AFFINE_PIT_ROOT_SEED_FILE'))
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    if not args.root_seed_file:
        print('refused: set AFFINE_PIT_ROOT_SEED_FILE or --root-seed-file (never embed)', file=sys.stderr)
        return 2
    if len(args.host_pubkey_hex) != 64:
        print('host-pubkey-hex must be 64 hex chars', file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists() and not args.force:
        print('refused: %s exists (pass --force to overwrite)' % out, file=sys.stderr)
        return 2

    root = _load_root_signing_key(args.root_seed_file)
    root_pk = bytes(root.verify_key).hex()
    if root_pk == PLACEHOLDER_ROOT_PUBKEY_HEX:
        print('refused: placeholder ROOT key', file=sys.stderr)
        return 2
    now = time.time()
    cert = sign_host_key_cert(
        root_signing_key=root,
        host_pubkey_hex=args.host_pubkey_hex,
        issued_at_unix=now,
        expires_at_unix=now + float(args.ttl_days) * 86400.0,
        host_id=args.host_id,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    tmp.write_text(json.dumps(cert, indent=2, sort_keys=True) + '\n')
    os.chmod(tmp, 0o644)
    os.replace(tmp, out)
    # Public outputs only — never the seed.
    print(json.dumps({
        'ok': True,
        'out': str(out),
        'root_pubkey_hex': root_pk,
        'host_pubkey_hex': args.host_pubkey_hex,
        'expires_at_unix': cert['cert']['expires_at_unix'],
        'bootstrap_override': False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
