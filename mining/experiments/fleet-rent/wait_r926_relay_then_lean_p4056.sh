#!/usr/bin/env bash
# p4056: re-arm wait→lean after p4053 died on SSH timeout (set -e). Tolerate ssh blips.
set -uo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r926_relay_then_lean_p4056.log
: >"$LOG"
log(){ echo "[p4056-r926-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=15)
DST_HOST=95.133.252.28; DST_PORT=40298
LEAN_LOCAL=$ROOT/experiments/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh

log "wait for SIZE_OK / r926_scp_ready.done on crown"
ready=0
for i in $(seq 1 720); do
  if ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "test -f /root/logs/r926_scp_ready.done" 2>/dev/null; then
    stamp=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "cat /root/logs/r926_scp_ready.done" 2>/dev/null || true)
    log "STAMP seen=$stamp poll=$i"
    ready=1
    break
  fi
  ok=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
    'n=$(ls /tmp/r926_merged/model-*-of-00016.safetensors 2>/dev/null | wc -l); t=$(ls /tmp/r926_merged/*.tmp 2>/dev/null | wc -l); echo $n:$t' 2>/dev/null || echo "sshfail")
  if [[ "$ok" == "sshfail" ]]; then
    log "ssh blip poll=$i — retry"
    sleep 10
    continue
  fi
  n=${ok%%:*}; t=${ok##*:}
  if [[ "$n" == "16" && "$t" == "0" ]]; then
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
      "echo '$stamp' > /root/logs/r926_scp_ready.done && echo SIZE_OK_stamp:$stamp" 2>/dev/null || true
    log "SIZE_OK stamped stamp=$stamp poll=$i"
    ready=1
    break
  fi
  if (( i % 6 == 0 )); then log "poll=$i status=$ok"; fi
  sleep 10
done
[[ "$ready" == "1" ]] || { log "FATAL no stamp after polls"; exit 1; }

scp "${SSH_OPTS[@]}" -P "$DST_PORT" "$LEAN_LOCAL" \
  "root@$DST_HOST:/root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh"

ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr /root/logs
chmod +x /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh
if [[ -d /root/.triton/cache/chall_r928 ]]; then
  rm -rf /root/.triton/cache/chall_r926
  cp -a /root/.triton/cache/chall_r928 /root/.triton/cache/chall_r926
fi
if [[ -f /root/logs/p4056_r926_lean_n80.outer.pid ]]; then
  opid=$(cat /root/logs/p4056_r926_lean_n80.outer.pid)
  if kill -0 "$opid" 2>/dev/null; then
    echo already_running:$opid
    exit 0
  fi
fi
# also skip if p4053 lean already alive
if [[ -f /root/logs/p4053_r926_lean_n80.outer.pid ]]; then
  opid=$(cat /root/logs/p4053_r926_lean_n80.outer.pid)
  if kill -0 "$opid" 2>/dev/null; then
    echo already_running_p4053:$opid
    exit 0
  fi
fi
nohup bash /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh \
  >/root/logs/p4056_r926_lean_n80.outer.nohup 2>&1 &
echo $! >/root/logs/p4056_r926_lean_n80.outer.pid
echo launched:$(cat /root/logs/p4056_r926_lean_n80.outer.pid)
REMOTE
log "lean armed"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/wait_r926_relay_then_lean_p4056.done"
