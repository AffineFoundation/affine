#!/usr/bin/env bash
# p4057: re-arm wait→lean after p4052 wait stuck on SSH timeouts (set -e). Tolerate ssh blips + SIZE_OK self-stamp.
set -uo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r934_relay_then_lean_p4057.log
: >"$LOG"
log(){ echo "[p4057-r934-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=15)
DST_HOST=95.133.252.28; DST_PORT=40298
LEAN_LOCAL=$ROOT/experiments/r934-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus67_p4052.sh

log "wait for SIZE_OK / r934_scp_ready.done on crown"
ready=0
for i in $(seq 1 720); do
  if ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "test -f /root/logs/r934_scp_ready.done" 2>/dev/null; then
    stamp=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "cat /root/logs/r934_scp_ready.done" 2>/dev/null || true)
    log "STAMP seen=$stamp poll=$i"
    ready=1
    break
  fi
  ok=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
    'n=$(ls /tmp/r934_merged/model-*-of-00016.safetensors 2>/dev/null | wc -l); t=$(ls /tmp/r934_merged/*.tmp 2>/dev/null | wc -l); v=$(test -f /tmp/r934_merged/model-visual-restored.safetensors && echo 1 || echo 0); echo $n:$t:$v' 2>/dev/null || echo "sshfail")
  if [[ "$ok" == "sshfail" ]]; then
    log "ssh blip poll=$i — retry"
    sleep 10
    continue
  fi
  n=${ok%%:*}; rest=${ok#*:}; t=${rest%%:*}; v=${rest##*:}
  if [[ "$n" == "16" && "$t" == "0" && "$v" == "1" ]]; then
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
      "echo '$stamp' > /root/logs/r934_scp_ready.done && echo SIZE_OK_stamp:$stamp" 2>/dev/null || true
    log "SIZE_OK stamped stamp=$stamp poll=$i"
    ready=1
    break
  fi
  if (( i % 6 == 0 )); then log "poll=$i status=$ok"; fi
  sleep 10
done
[[ "$ready" == "1" ]] || { log "FATAL no stamp after polls"; exit 1; }

scp "${SSH_OPTS[@]}" -P "$DST_PORT" "$LEAN_LOCAL" \
  "root@$DST_HOST:/root/mining_src/r934-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus67_p4052.sh"

ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r934-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr /root/logs
chmod +x /root/mining_src/r934-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus67_p4052.sh
if [[ -d /root/.triton/cache/chall_r928 ]]; then
  rm -rf /root/.triton/cache/chall_r934
  cp -a /root/.triton/cache/chall_r928 /root/.triton/cache/chall_r934
fi
# skip if already armed
for pf in /root/logs/p4057_r934_lean_n80.outer.pid /root/logs/p4052_r934_lean_n80.outer.pid; do
  if [[ -f "$pf" ]]; then
    opid=$(cat "$pf")
    if kill -0 "$opid" 2>/dev/null; then
      echo already_running:$opid:$pf
      exit 0
    fi
  fi
done
nohup bash /root/mining_src/r934-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus67_p4052.sh \
  >/root/logs/p4057_r934_lean_n80.outer.nohup 2>&1 &
echo $! >/root/logs/p4057_r934_lean_n80.outer.pid
echo launched:$(cat /root/logs/p4057_r934_lean_n80.outer.pid)
REMOTE
log "lean armed"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/wait_r934_relay_then_lean_p4057.done"
