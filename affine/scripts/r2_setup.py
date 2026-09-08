"""One-shot, idempotent R2 infrastructure for the private-submission flow.

    source .venv/bin/activate && source .env
    python affine/scripts/r2_setup.py [--zone-id <affine.io zone>] [--apply]

Creates the three buckets from affine.toml [submission.r2], attaches the
custom public domains (public models + dash/mailbox) to the affine.io zone,
sets lifecycle rules (private registration prefixes expire after
private_retention_days; mailbox blobs after 8 days), and mints the read-only
eval-pod token. Prints the .env lines the validator still needs
(AFFINE_EVAL_R2_* and, if unset, a fresh AFFINE_MAILBOX_SIGNING_SEED).
Without --apply it only reports what it would do.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affine import r2  # noqa: E402
from affine.config import load_config  # noqa: E402
from affine.registrations import signer_from_seed  # noqa: E402

EVAL_TOKEN_NAME = "affine-eval-readonly"
MAILBOX_RETENTION_DAYS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--zone-id", default=os.environ.get("CLOUDFLARE_ZONE_ID", ""),
                    help="Cloudflare zone id for the public domains' apex")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rotate-eval-token", action="store_true",
                    help="revoke + recreate the read-only eval token")
    args = ap.parse_args()

    cfg = load_config()
    r2c = cfg.submission.r2
    sec = cfg.secrets
    admin = r2.CloudflareR2Admin(sec.cloudflare_account_id, sec.cloudflare_api_token)
    print(f"account {sec.cloudflare_account_id}  endpoint {sec.r2_endpoint}")
    print(f"buckets: private={r2c.private_bucket} public={r2c.public_bucket} "
          f"dash={r2c.dash_bucket}")
    if len({r2c.private_bucket, r2c.public_bucket, r2c.dash_bucket}) != 3:
        raise SystemExit("the three bucket names must be distinct")

    have = set(admin.list_buckets())
    for name in (r2c.private_bucket, r2c.public_bucket, r2c.dash_bucket):
        if name in have:
            print(f"  bucket {name}: exists")
        elif args.apply:
            admin.ensure_bucket(name)
            print(f"  bucket {name}: CREATED")
        else:
            print(f"  bucket {name}: would create")

    domains = {
        r2c.public_bucket: r2c.public_models_base_url.removeprefix("https://"),
        r2c.dash_bucket: r2c.mailbox_base_url.removeprefix("https://"),
    }
    for bucket, host in domains.items():
        if not host or bucket not in have and not args.apply:
            continue
        try:
            current = {d.get("domain") for d in admin.list_custom_domains(bucket)}
        except r2.CloudflareError as e:
            print(f"  domain {host} → {bucket}: cannot list ({e})")
            continue
        if host in current:
            print(f"  domain {host} → {bucket}: attached")
        elif args.apply:
            if not args.zone_id:
                raise SystemExit("--zone-id (or CLOUDFLARE_ZONE_ID) required to attach domains")
            admin.add_custom_domain(bucket, host, args.zone_id)
            print(f"  domain {host} → {bucket}: ATTACHED (DNS + cert may take a few min)")
        else:
            print(f"  domain {host} → {bucket}: would attach")

    lifecycle = {
        r2c.private_bucket: [{
            "id": "expire-registrations", "enabled": True,
            "conditions": {"prefix": "models/registrations/"},
            "deleteObjectsTransition": {
                "condition": {"type": "Age",
                              "maxAge": r2c.private_retention_days * 86400}},
            "abortMultipartUploadsTransition": {
                "condition": {"type": "Age", "maxAge": 7 * 86400}},
        }],
        r2c.dash_bucket: [{
            "id": "expire-mailbox", "enabled": True,
            "conditions": {"prefix": "mailbox/"},
            "deleteObjectsTransition": {
                "condition": {"type": "Age",
                              "maxAge": MAILBOX_RETENTION_DAYS * 86400}},
        }],
        r2c.public_bucket: [{
            "id": "abort-stale-multipart", "enabled": True,
            "conditions": {"prefix": ""},
            "abortMultipartUploadsTransition": {
                "condition": {"type": "Age", "maxAge": 7 * 86400}},
        }],
    }
    for bucket, rules in lifecycle.items():
        if args.apply:
            admin.put_lifecycle(bucket, rules)
            print(f"  lifecycle {bucket}: set ({', '.join(r['id'] for r in rules)})")
        else:
            print(f"  lifecycle {bucket}: would set ({', '.join(r['id'] for r in rules)})")

    env_lines: list[str] = []
    existing = admin.list_tokens(EVAL_TOKEN_NAME)
    if existing and not args.rotate_eval_token:
        print(f"  eval token {EVAL_TOKEN_NAME}: exists (id {existing[0]['id']}); "
              "--rotate-eval-token to recreate")
    elif args.apply:
        for t in existing:
            admin.delete_token(str(t["id"]))
        # Read-only over BOTH model buckets: two policies in one token.
        perm = admin._permission_id(r2.PERM_BUCKET_ITEM_READ)
        res = admin._call("POST", "/tokens", {
            "name": EVAL_TOKEN_NAME,
            "policies": [{
                "effect": "allow",
                "resources": {admin._bucket_resource(b): "*"
                              for b in (r2c.private_bucket, r2c.public_bucket)},
                "permission_groups": [{"id": perm}],
            }],
        })
        token_id, value = str(res["id"]), str(res["value"])
        env_lines += [f"AFFINE_EVAL_R2_ACCESS_KEY_ID={token_id}",
                      f"AFFINE_EVAL_R2_SECRET_ACCESS_KEY="
                      f"{hashlib.sha256(value.encode()).hexdigest()}"]
        print(f"  eval token {EVAL_TOKEN_NAME}: CREATED (id {token_id})")
    else:
        print(f"  eval token {EVAL_TOKEN_NAME}: would create")

    if sec.mailbox_signing_seed:
        signer = signer_from_seed(sec.mailbox_signing_seed)
        print(f"  mailbox signer: {signer.ss58_address} (from AFFINE_MAILBOX_SIGNING_SEED)")
    else:
        seed = secrets.token_hex(32)
        signer = signer_from_seed(seed)
        env_lines.append(f"AFFINE_MAILBOX_SIGNING_SEED={seed}")
        print(f"  mailbox signer: NEW {signer.ss58_address}")

    if env_lines:
        print("\nadd to .env:")
        for line in env_lines:
            print("  " + line)


if __name__ == "__main__":
    main()
