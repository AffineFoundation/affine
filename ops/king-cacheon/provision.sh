#!/bin/bash
# Operator side: push bootstrap_h100.sh + env to the Cacheon king pod and
# launch it. Reads the current king digest from the public snapshot and the
# winner's content hash / disclosure time from dash.cacheon.ai.
#
#   provision.sh root@HOST -p PORT [RESERVATION_ID]
#
# Pod: Lium 1×H100 80 GB (SXM) with the Pytorch (Cuda + DinD) template —
# the arena image is built and run with docker on the pod.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SSH_TARGET=${1:?"root@HOST"}; shift
PORT_FLAG=${1:?"-p"}; PORT=${2:?PORT}; shift 2
RESERVATION_ID=${1:-0309b8665292eeedfedf3823abe09ddef570e4b878c2dc816f8e414280250c3e}
CACHEON_REV=${CACHEON_REV:-8c186244fad6bacfc04800d15021b1989efa5622}
PUBLIC_PORT=${PUBLIC_PORT:-20000}
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o LogLevel=ERROR)

king=$(curl -fsS https://affine.io/api/v1/snapshot | python3 -c "import sys,json; print(json.load(sys.stdin)['king']['revision'])")
read -r content_hash release_at < <(curl -fsS -A "Mozilla/5.0" \
  "https://dash.cacheon.ai/api/submissions/$RESERVATION_ID?arena=qwen" | python3 -c "
import sys, json; d = json.load(sys.stdin); print(d['content_hash'], d['bundle_visibility']['release_at'])")
echo "king=$king reservation=$RESERVATION_ID content_hash=$content_hash release_at=$release_at"

scp "${SSH_OPTS[@]}" -P "$PORT" "$HERE/bootstrap_h100.sh" "$SSH_TARGET:/tmp/bootstrap_h100.sh"
printf 'KING_DIGEST=%s\nRESERVATION_ID=%s\nCONTENT_HASH=%s\nRELEASE_AT=%s\nCACHEON_REV=%s\nPUBLIC_PORT=%s\n' \
  "$king" "$RESERVATION_ID" "$content_hash" "$release_at" "$CACHEON_REV" "$PUBLIC_PORT" | \
  ssh "${SSH_OPTS[@]}" -p "$PORT" "$SSH_TARGET" \
    'mkdir -p /root/cacheon-king/logs && cat > /root/cacheon-king/env && \
     mv /tmp/bootstrap_h100.sh /root/cacheon-king/bootstrap.sh && chmod +x /root/cacheon-king/bootstrap.sh && \
     (setsid nohup bash /root/cacheon-king/bootstrap.sh >> /root/cacheon-king/logs/bootstrap.log 2>&1 < /dev/null &) && echo LAUNCHED'
echo "follow: ssh $SSH_TARGET -p $PORT tail -f /root/cacheon-king/logs/bootstrap.log"
