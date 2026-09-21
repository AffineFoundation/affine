#!/bin/bash
# Publish a healthy eval pod's uv download cache as the wheelhouse tarball
# that evalsrv/bootstrap.sh restores on fresh pods (2026-09-21).
#
#   ops/eval_wheelhouse/publish_uv_cache.sh            # eval pod from state.json
#   ops/eval_wheelhouse/publish_uv_cache.sh root@HOST -p PORT
#
# Streams `tar | zstd` off the pod over ssh straight into `aws s3 cp` on the
# affine-data bucket (public at https://data.affine.io/wheelhouse/), keyed
# by python version + sha256(affine/pyproject.toml)[:8] — the same key the
# bootstrap computes. Credentials: DATA_R2_ACCESS_KEY_ID /
# DATA_R2_SECRET_ACCESS_KEY (+ DATA_R2_ENDPOINT) from the repo .env, as in
# ops/corpus_build.py. Idempotent: an existing object for the key is kept
# unless --force.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi
if [[ $# -ge 1 ]]; then
  SSH_TARGET="$*"
else
  SSH_TARGET=$(python3 -c "import json;print(json.load(open('$REPO/affine/state/state.json'))['eval_machine'].get('ssh',''))")
fi
[[ -n "$SSH_TARGET" ]] || { echo "no eval machine ssh target" >&2; exit 1; }
HOST=${SSH_TARGET%% *}; PORT=${SSH_TARGET##* }

val() { grep -E "^(export )?$1=" "$REPO/.env" 2>/dev/null | tail -1 | sed -E 's/^(export )?[^=]+=//; s/^"//; s/"$//'; }
export AWS_ACCESS_KEY_ID=${DATA_R2_ACCESS_KEY_ID:-$(val DATA_R2_ACCESS_KEY_ID)}
export AWS_SECRET_ACCESS_KEY=${DATA_R2_SECRET_ACCESS_KEY:-$(val DATA_R2_SECRET_ACCESS_KEY)}
ENDPOINT=${DATA_R2_ENDPOINT:-$(val DATA_R2_ENDPOINT)}
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" && -n "$ENDPOINT" ]] || { echo "DATA_R2_* missing" >&2; exit 1; }
export AWS_DEFAULT_REGION=auto

KEY="py312-$(sha256sum "$REPO/affine/pyproject.toml" | cut -c1-8)"
OBJ="s3://affine-data/wheelhouse/uv-cache-${KEY}.tar.zst"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -p "$PORT" "$HOST")

if [[ $FORCE -eq 0 ]] && aws --endpoint-url "$ENDPOINT" s3 ls "$OBJ" >/dev/null 2>&1; then
  echo "wheelhouse already published for $KEY: $OBJ (use --force to replace)"; exit 0
fi
# The pod's pyproject must match ours, or the key lies.
remote_key=$("${SSH[@]}" "sha256sum /root/affine/pyproject.toml | cut -c1-8")
[[ "$remote_key" == "${KEY#py312-}" ]] || { echo "pod pyproject sha $remote_key != local ${KEY#py312-}" >&2; exit 1; }
"${SSH[@]}" "command -v zstd >/dev/null || apt-get install -y -qq zstd >/dev/null; du -sh /root/.cache/uv | cut -f1" | sed 's/^/pod uv cache: /'
echo "publishing $OBJ ..."
t0=$(date +%s)
"${SSH[@]}" "cd /root/.cache/uv && tar -cf - --exclude=.affine-wheelhouse . | zstd -T0 -3" \
  | aws --endpoint-url "$ENDPOINT" s3 cp - "$OBJ" --expected-size 20000000000 --only-show-errors
size=$(aws --endpoint-url "$ENDPOINT" s3 ls "$OBJ" | awk '{print $3}')
echo "published $OBJ ($size bytes) in $(( $(date +%s) - t0 ))s -> https://data.affine.io/wheelhouse/uv-cache-${KEY}.tar.zst"
