#!/usr/bin/env bash
# p4060 outer: host-relay R925 → R252 then lean+n80 on GPUs 6,7 :8002
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r925_to_r252_p4060.outer.nohup
exec >>"$LOG" 2>&1
echo "[p4060-outer] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"

RELAY=$ROOT/experiments/fleet-rent/host_relay_r925_to_r252_p4060.sh
LEAN_LOCAL=$ROOT/experiments/r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4060.sh
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
DST_HOST=38.127.229.127; DST_PORT=40299

bash "$RELAY"
test -f "$LOGDIR/host_relay_r925_to_r252_p4060.done"
echo "[p4060-outer] $(date -u +%Y-%m-%dT%H:%M:%SZ) RELAY_DONE"

# upload lean via stdin (scp can be flaky)
ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
  "mkdir -p /root/mining_src/r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr"
cat "$LEAN_LOCAL" | ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
  "cat > /root/mining_src/r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4060.sh && chmod +x /root/mining_src/r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4060.sh"
echo "[p4060-outer] $(date -u +%Y-%m-%dT%H:%M:%SZ) LEAN_UPLOADED"

# verify SIZE_OK on dst then launch
ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
n=$(ls /tmp/r925_merged/model-*-of-*.safetensors | wc -l)
test -f /root/logs/r925_scp_ready.done
test "$n" -ge 16
test -f /tmp/r925_merged/config.json
echo "SIZE_OK shards=$n stamp=$(cat /root/logs/r925_scp_ready.done)"
nohup bash /root/mining_src/r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4060.sh \
  > /root/logs/r925_lean_n80_outer.nohup 2>&1 &
echo $! > /root/logs/r925_lean_n80_outer.pid
echo "LEAN_LAUNCHED pid=$(cat /root/logs/r925_lean_n80_outer.pid)"
REMOTE

echo "[p4060-outer] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE arm lean"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGDIR/host_relay_r925_to_r252_p4060.outer.done"
