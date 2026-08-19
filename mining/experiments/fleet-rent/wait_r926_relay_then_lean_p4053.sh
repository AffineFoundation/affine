#!/usr/bin/env bash
# p4053: after R926 SIZE_OK stamp on crown, launch lean chall+n80 on GPUs 1,3 :8003.
# Gap: host_relay_r926 stamps r926_scp_ready.done but nothing was waiting to lean.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r926_relay_then_lean_p4053.log
: >"$LOG"
log(){ echo "[p4053-r926-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes)
DST_HOST=95.133.252.28; DST_PORT=40298
LEAN_LOCAL=$ROOT/experiments/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh

log "wait for /root/logs/r926_scp_ready.done on crown"
for i in $(seq 1 720); do
  if ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "test -f /root/logs/r926_scp_ready.done"; then
    stamp=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "cat /root/logs/r926_scp_ready.done")
    log "STAMP seen=$stamp poll=$i"
    break
  fi
  # also SIZE_OK without stamp (p4049 lesson): 16 shards + 0 tmps
  ok=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
    'n=$(ls /tmp/r926_merged/model-*-of-00016.safetensors 2>/dev/null | wc -l); t=$(ls /tmp/r926_merged/*.tmp 2>/dev/null | wc -l); echo $n:$t')
  n=${ok%%:*}; t=${ok##*:}
  if [[ "$n" == "16" && "$t" == "0" ]]; then
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
      "echo '$stamp' > /root/logs/r926_scp_ready.done && echo SIZE_OK_stamp:$stamp"
    log "SIZE_OK stamped locally stamp=$stamp poll=$i"
    break
  fi
  sleep 10
done
ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "test -f /root/logs/r926_scp_ready.done" || { log "FATAL no stamp"; exit 1; }

scp "${SSH_OPTS[@]}" -P "$DST_PORT" "$LEAN_LOCAL" \
  "root@$DST_HOST:/root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh"

ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr /root/logs
chmod +x /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh
# prefer chall_r928 seed into chall_r926 before lean (p4051 lesson)
if [[ -d /root/.triton/cache/chall_r928 ]]; then
  rm -rf /root/.triton/cache/chall_r926
  cp -a /root/.triton/cache/chall_r928 /root/.triton/cache/chall_r926
fi
# do not relaunch if already running
if [[ -f /root/logs/p4053_r926_lean_n80.outer.pid ]]; then
  opid=$(cat /root/logs/p4053_r926_lean_n80.outer.pid)
  if kill -0 "$opid" 2>/dev/null; then
    echo already_running:$opid
    exit 0
  fi
fi
nohup bash /root/mining_src/r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus13_p4051.sh \
  >/root/logs/p4053_r926_lean_n80.outer.nohup 2>&1 &
echo $! >/root/logs/p4053_r926_lean_n80.outer.pid
echo launched:$(cat /root/logs/p4053_r926_lean_n80.outer.pid)
REMOTE
log "lean armed"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/wait_r926_relay_then_lean_p4053.done"
