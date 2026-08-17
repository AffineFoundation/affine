#!/usr/bin/env bash
# p3687: R633 unblock — kill path was stuck on wrong R634 done-file name; R634+R647 both
# REFUTED; zesty 4,5 occupied by R662 TRAIN. Solo brave→golden SCP then chall 4,5/:8003 v4.
# Never dual-pipe. Never pkill -f.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r633-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megaextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r633_p3687.log
: >"$LOG"
log() { echo "[p3687-r633] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r633-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3687.sh
WAIT_LOCAL=$EXP/wait_r633_scp_king_then_chall_p3687.sh

log "verify no other brave tar uplink"
bt=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep -E 'tar cf - r6[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${bt:-99}" -eq 0 ]] || { log "FATAL brave already tar-piping (count=$bt)"; exit 1; }

log "verify brave R633 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r633_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r633 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "refresh lean+wait on golden; clear stale r633 dest"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "chmod +x $REMOTE_DIR/*.sh"

ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'r633_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3687] kill hung r633 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
# purge leftover REFUTE dests if present (keep r637 / r647 for Triton seed)
for d in /tmp/r629_merged /tmp/r646_merged /tmp/r648_merged; do
  if [[ -d "$d" ]]; then
    echo "[p3687] purge $d"
    rm -rf "$d"
  fi
done
rm -rf /tmp/r633_merged
rm -f /root/logs/r633_scp_ready.done /root/logs/r633_chall_n80_launched.p3687 \
  /root/logs/r633_chall_n80_launched.p3623
mkdir -p /tmp/r633_merged
if [[ -f /root/logs/p3687_r633_wait.pid ]] && kill -0 "$(cat /root/logs/p3687_r633_wait.pid)" 2>/dev/null; then
  echo "waiter_alive pid=$(cat /root/logs/p3687_r633_wait.pid)"
else
  nohup bash /root/mining_src/r633-chall/wait_r633_scp_king_then_chall_p3687.sh \
    >/root/logs/p3687_r633_wait.nohup 2>&1 &
  echo $! >/root/logs/p3687_r633_wait.pid
  echo "waiter_started pid=$(cat /root/logs/p3687_r633_wait.pid)"
fi
code=$(curl -s -o /tmp/king_models_p3687_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3687_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3687-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/swap_king_reign34_golden_p3687.done
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe brave→golden (~66G R633) solo"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r633_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r633_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r633_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r633_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r633_merged"
rc_b=${PIPESTATUS[0]}
rc_g=${PIPESTATUS[1]}
set -e
log "pipe exit brave=$rc_b golden=$rc_g"
[[ "$rc_b" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+v4-n80 on 4,5/:8003"
log "DONE"
