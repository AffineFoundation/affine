#!/bin/bash
# Publish a healthy eval pod's uv download cache as the wheelhouse tarball
# that evalsrv/bootstrap.sh restores on fresh pods (2026-09-21).
#
#   ops/eval_wheelhouse/publish_uv_cache.sh            # eval pod from state.json
#   ops/eval_wheelhouse/publish_uv_cache.sh root@HOST -p PORT
#
# Streams `tar | zstd` off the pod over ssh straight into a boto3 multipart
# upload on the affine-data bucket (public at https://data.affine.io/
# wheelhouse/), keyed by python version + sha256(affine/pyproject.toml)[:8]
# — the same key the bootstrap computes. Credentials: DATA_R2_ACCESS_KEY_ID /
# DATA_R2_SECRET_ACCESS_KEY / DATA_R2_ENDPOINT from the repo .env, as in
# ops/corpus_build.py. Idempotent: an existing object for the key is kept
# unless --force.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi
if [[ $# -ge 1 ]]; then
  SSH_TARGET="$*"
else
  SSH_TARGET=$("$PY" -c "import json;print(json.load(open('$REPO/affine/state/state.json'))['eval_machine'].get('ssh',''))")
fi
[[ -n "$SSH_TARGET" ]] || { echo "no eval machine ssh target" >&2; exit 1; }
HOST=${SSH_TARGET%% *}; PORT=${SSH_TARGET##* }

val() { grep -E "^(export )?$1=" "$REPO/.env" 2>/dev/null | tail -1 | sed -E 's/^(export )?[^=]+=//; s/^"//; s/"$//'; }
export DATA_R2_ACCESS_KEY_ID=${DATA_R2_ACCESS_KEY_ID:-$(val DATA_R2_ACCESS_KEY_ID)}
export DATA_R2_SECRET_ACCESS_KEY=${DATA_R2_SECRET_ACCESS_KEY:-$(val DATA_R2_SECRET_ACCESS_KEY)}
export DATA_R2_ENDPOINT=${DATA_R2_ENDPOINT:-$(val DATA_R2_ENDPOINT)}
[[ -n "$DATA_R2_ACCESS_KEY_ID" && -n "$DATA_R2_SECRET_ACCESS_KEY" && -n "$DATA_R2_ENDPOINT" ]] || { echo "DATA_R2_* missing" >&2; exit 1; }

KEY="py312-$(sha256sum "$REPO/affine/pyproject.toml" | cut -c1-8)"
OBJKEY="wheelhouse/uv-cache-${KEY}.tar.zst"
export WH_KEY="$OBJKEY" WH_FORCE="$FORCE"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -p "$PORT" "$HOST")

if [[ $FORCE -eq 0 ]] && "$PY" - <<'PY'
import os, boto3
s3 = boto3.client("s3", endpoint_url=os.environ["DATA_R2_ENDPOINT"], aws_access_key_id=os.environ["DATA_R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["DATA_R2_SECRET_ACCESS_KEY"], region_name="auto")
try:
    h = s3.head_object(Bucket="affine-data", Key=os.environ["WH_KEY"]); print(f"already published: {os.environ['WH_KEY']} ({h['ContentLength']} bytes); use --force to replace"); raise SystemExit(0)
except s3.exceptions.ClientError:
    raise SystemExit(1)
PY
then exit 0; fi

remote_key=$("${SSH[@]}" "sha256sum /root/affine/pyproject.toml | cut -c1-8")
[[ "$remote_key" == "${KEY#py312-}" ]] || { echo "pod pyproject sha $remote_key != local ${KEY#py312-}" >&2; exit 1; }
"${SSH[@]}" "command -v zstd >/dev/null || { apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd >/dev/null 2>&1; }; command -v zstd >/dev/null || exit 42; du -sh /root/.cache/uv | cut -f1" | sed 's/^/pod uv cache: /' \
  || { echo "zstd unavailable on the pod; nothing uploaded" >&2; exit 1; }
echo "publishing s3://affine-data/$OBJKEY ..."
t0=$(date +%s)
"${SSH[@]}" "cd /root/.cache/uv && tar -cf - --exclude=.affine-wheelhouse . | zstd -T0 -3" \
  | "$PY" - <<'PY'
import os, sys, boto3
from boto3.s3.transfer import TransferConfig
s3 = boto3.client("s3", endpoint_url=os.environ["DATA_R2_ENDPOINT"], aws_access_key_id=os.environ["DATA_R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["DATA_R2_SECRET_ACCESS_KEY"], region_name="auto")
cfg = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4)
s3.upload_fileobj(sys.stdin.buffer, "affine-data", os.environ["WH_KEY"], ExtraArgs={"ContentType": "application/zstd"}, Config=cfg)
h = s3.head_object(Bucket="affine-data", Key=os.environ["WH_KEY"])
if h["ContentLength"] < 100 * 1024 * 1024:
    s3.delete_object(Bucket="affine-data", Key=os.environ["WH_KEY"])
    raise SystemExit(f"upload too small ({h['ContentLength']} bytes) — deleted; check the pod-side tar")
print(f"uploaded {h['ContentLength']} bytes")
PY
echo "published in $(( $(date +%s) - t0 ))s -> https://data.affine.io/${OBJKEY}"
