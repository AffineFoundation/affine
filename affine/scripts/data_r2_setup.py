"""One-shot, idempotent R2 infrastructure for the trace-first corpus.

    source .venv/bin/activate && source .env
    python affine/scripts/data_r2_setup.py [--zone-id <affine.io zone>] [--apply]

Creates the [data_r2] bucket from affine.toml, attaches its public custom
domain (data.affine.io) to the affine.io zone, sets a stale-multipart
lifecycle rule, and mints two bucket-scoped S3 credentials:

  affine-data-writer     datagen pod — puts trace chunks + trace manifests
  affine-data-publisher  validator box — publishes views + corpus manifests

Prints the env lines each side needs. Token values are shown once; the
script never stores them. Without --apply it only reports what it would do.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affine import r2  # noqa: E402
from affine.config import load_config  # noqa: E402

WRITER_TOKEN = "affine-data-writer"
PUBLISHER_TOKEN = "affine-data-publisher"


def _token(admin: r2.CloudflareR2Admin, name: str, bucket: str, *,
           apply: bool, rotate: bool) -> list[str]:
    existing = admin.list_tokens(name)
    if existing and not rotate:
        print(f"  token {name}: exists (id {existing[0]['id']}); --rotate to recreate")
        return []
    if not apply:
        print(f"  token {name}: would create (bucket-scoped object write)")
        return []
    for t in existing:
        admin.delete_token(str(t["id"]))
    token_id, value = admin.create_bucket_token(name, bucket, write=True)
    print(f"  token {name}: CREATED (id {token_id})")
    prefix = "DATA_R2" if name == PUBLISHER_TOKEN else "ROLLOUTS_R2"
    return [f"{prefix}_ACCESS_KEY_ID={token_id}",
            f"{prefix}_SECRET_ACCESS_KEY={hashlib.sha256(value.encode()).hexdigest()}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--zone-id", default=os.environ.get("CLOUDFLARE_ZONE_ID", ""))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rotate", action="store_true",
                    help="revoke + recreate both bucket tokens")
    args = ap.parse_args()

    cfg = load_config()
    dr = cfg.data_r2
    sec = cfg.secrets
    admin = r2.CloudflareR2Admin(sec.cloudflare_account_id, sec.cloudflare_api_token)
    host = dr["public_base_url"].removeprefix("https://")
    print(f"account {sec.cloudflare_account_id}  bucket {dr['bucket']}  domain {host}")

    have = set(admin.list_buckets())
    if dr["bucket"] in have:
        print(f"  bucket {dr['bucket']}: exists")
    elif args.apply:
        admin.ensure_bucket(dr["bucket"])
        print(f"  bucket {dr['bucket']}: CREATED")
    else:
        print(f"  bucket {dr['bucket']}: would create")

    if dr["bucket"] in have or args.apply:
        current = {d.get("domain") for d in admin.list_custom_domains(dr["bucket"])}
        if host in current:
            print(f"  domain {host}: attached")
        elif args.apply:
            if not args.zone_id:
                raise SystemExit("--zone-id (or CLOUDFLARE_ZONE_ID) required")
            admin.add_custom_domain(dr["bucket"], host, args.zone_id)
            print(f"  domain {host}: ATTACHED (DNS + cert may take a few min)")
        else:
            print(f"  domain {host}: would attach")
        managed = admin.get_managed_public_domain(dr["bucket"])
        if managed.get("enabled"):
            print("  r2.dev public domain: enabled — disabling (custom domain only)")
            if args.apply:
                admin.set_managed_public_domain(dr["bucket"], False)

    rules = [{
        "id": "abort-stale-multipart", "enabled": True,
        "conditions": {"prefix": ""},
        "abortMultipartUploadsTransition": {
            "condition": {"type": "Age", "maxAge": 7 * 86400}},
    }]
    if args.apply:
        admin.put_lifecycle(dr["bucket"], rules)
        print("  lifecycle: set (abort-stale-multipart)")
        admin.put_public_read_cors(dr["bucket"])
        print("  cors: set (GET/HEAD from any origin)")
    else:
        print("  lifecycle: would set (abort-stale-multipart)")
        print("  cors: would set (GET/HEAD from any origin)")

    env_lines: list[str] = []
    for name in (WRITER_TOKEN, PUBLISHER_TOKEN):
        env_lines += _token(admin, name, dr["bucket"], apply=args.apply,
                            rotate=args.rotate)
    if env_lines:
        print("\nAdd to the matching side's env (ROLLOUTS_R2_* -> pod "
              ".rollouts_env, DATA_R2_* -> validator .env):")
        print(f"DATA_R2_ENDPOINT={sec.r2_endpoint}")
        for line in env_lines:
            print(line)


if __name__ == "__main__":
    main()
