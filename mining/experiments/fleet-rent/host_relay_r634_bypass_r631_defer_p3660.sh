#!/usr/bin/env bash
# p3660: R634 RETARGET — skip R631 gate (R631 DEFER). R641 SCP_READY + zesty 6,7 free.
# Solo host-relay brave→zesty → chall 6,7/:8003. Never pkill -f.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r634-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megaextrasteps-ep2-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r634_p3660.log
: >"$LOG"
log() { echo "[p3660-r634] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=20 -o ServerAliveCountMax=12 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
ZESTY_HOST=86.38.182.95
ZESTY_PORT=20299
REMOTE_DIR=/root/mining_src/r634-chall
LEAN_LOCAL=$EXP/lean_chall_n80_zesty_gpus67_p3646.sh
WAIT_LOCAL=$EXP/wait_r634_scp_king_then_chall_p3646.sh

log "bypass R631 DEFER — verify R641 ready + GPUs 6,7 free + brave source"
r641=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
  'if [[ -f /root/logs/r641_scp_ready.done ]] && [[ -f /tmp/r641_merged/config.json ]]; then
     n=$(ls /tmp/r641_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ready:$n"
   else echo missing; fi' 2>/dev/null || echo sshfail)
log "r641=$r641"
[[ "$r641" == ready:* ]] || { log "FATAL R641 not ready"; exit 1; }

free67=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
  'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F", " "\$1==6 || \$1==7 {print}"' 2>/dev/null || true)
log "gpus67=$free67"
echo "$free67" | grep -q '6, 0' || { log "FATAL GPU6 not free"; exit 1; }
echo "$free67" | grep -q '7, 0' || { log "FATAL GPU7 not free"; exit 1; }

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r634_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r634 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait (6,7/:8003) to zesty"
ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "mkdir -p $REMOTE_DIR /root/logs"
scp "${SSH_OPTS[@]}" -P "$ZESTY_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$ZESTY_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "chmod +x $REMOTE_DIR/*.sh"

ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
# Kill hung r634 tar xf only (by PID scan, never pkill -f)
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'r634_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3660] kill hung r634 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
# Stop stale waiters by pidfile only
for pf in /root/logs/p3612_r634_wait.pid /root/logs/p3646_r634_wait.pid; do
  if [[ -f "$pf" ]]; then
    pid=$(cat "$pf" 2>/dev/null || true)
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[p3660] stop stale waiter pid=$pid ($pf)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
  fi
done
rm -rf /tmp/r634_merged
rm -f /root/logs/r634_scp_ready.done \
  /root/logs/r634_chall_n80_launched.p3612 \
  /root/logs/r634_chall_n80_launched.p3646 \
  /root/logs/r634_reign34_pipeline.done
mkdir -p /tmp/r634_merged
nohup bash /root/mining_src/r634-chall/wait_r634_scp_king_then_chall_p3646.sh \
  >/root/logs/p3660_r634_wait.nohup 2>&1 &
echo $! >/root/logs/p3660_r634_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3660_r634_wait.pid)"
code=$(curl -s -o /tmp/king_models_p3660_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3660_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3660-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
[[ -f /root/logs/swap_king_reign34_zesty_p3596.done ]] || date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/swap_king_reign34_zesty_p3596.done
df -h / | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
REMOTE

log "begin host-relay tar pipe brave→zesty (~66G R634) solo → chall 6,7/:8003"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r634_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r634_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r634_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r634_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r634_merged"
rc_b=${PIPESTATUS[0]}
rc_z=${PIPESTATUS[1]}
set -e
log "pipe exit brave=$rc_b zesty=$rc_z"
[[ "$rc_b" -eq 0 && "$rc_z" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+n80 on 6,7/:8003"
log "DONE"
