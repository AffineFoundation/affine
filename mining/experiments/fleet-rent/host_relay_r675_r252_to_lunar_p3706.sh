#!/usr/bin/env bash
# p3706: after R655 v4 REFUTE, arm R675 MERGE_DONE from R252 → lunar 4,5/:8003.
# Soft MidRank MidBeta SoftCtx UltraExtra ep3×LoLR. Never dual-pipe (R663 is crown→golden).
# Never pkill -f.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r675-r252-offline-dpo-hialpha-midrank-midbeta-softctx-ultraextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r675_p3706.log
: >"$LOG"
log() { echo "[p3706-r675] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
SRC_HOST=95.133.252.28
SRC_PORT=40299
LUNAR_HOST=150.136.46.118
LUNAR_PORT=20299
REMOTE_DIR=/root/mining_src/r675-chall
LEAN_LOCAL=$EXP/lean_chall_n80_lunar_gpus45_p3706.sh
WAIT_LOCAL=$EXP/wait_r675_scp_king_then_chall_p3706.sh

log "verify no other R252 tar uplink"
bt=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${bt:-99}" -eq 0 ]] || { log "FATAL R252 already tar-piping (count=$bt)"; exit 1; }

log "verify R252 R675 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
  "ls /tmp/r675_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "R252 r675 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to lunar; clear stale r655/r675 dest"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$LUNAR_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "chmod +x $REMOTE_DIR/*.sh"

ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
# kill hung r675/r655 tar xf only by PID scan (never pkill -f)
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r(655|675)_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3706] kill hung tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
# reap finished R655 chall :8003 by PID (sim already done)
for pidf in /root/logs/vllm_chall_r655.pid /root/logs/r655_sim_wvk7.pid /root/logs/p3688_r655_lean_outer.pid; do
  if [[ -f "$pidf" ]]; then
    pid=$(cat "$pidf" 2>/dev/null || true)
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[p3706] stop $pidf pid=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
  fi
done
# also kill any leftover r655 vllm on :8003
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "[p3706] kill leftover r655/argv pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r655_merged/ && !/awk/ {print $1}')
# free space for R675
rm -rf /tmp/r655_merged /tmp/r675_merged
rm -f /root/logs/r675_scp_ready.done /root/logs/r675_chall_n80_launched.p3706
mkdir -p /tmp/r675_merged
if [[ -f /root/logs/p3706_r675_wait.pid ]] && kill -0 "$(cat /root/logs/p3706_r675_wait.pid)" 2>/dev/null; then
  echo "waiter_alive pid=$(cat /root/logs/p3706_r675_wait.pid)"
else
  nohup bash /root/mining_src/r675-chall/wait_r675_scp_king_then_chall_p3706.sh \
    >/root/logs/p3706_r675_wait.nohup 2>&1 &
  echo $! >/root/logs/p3706_r675_wait.pid
  echo "waiter_started pid=$(cat /root/logs/p3706_r675_wait.pid)"
fi
code=$(curl -s -o /tmp/king_models_p3706_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3706_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3706-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/swap_king_reign34_lunar_p3706.done
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe R252→lunar (~67G R675) solo (crown R663→golden untouched)"
set +e
ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "cd /tmp && tar cf - r675_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r675_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r675_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r675_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r675_merged"
rc_s=${PIPESTATUS[0]}
rc_l=${PIPESTATUS[1]}
set -e
log "pipe exit R252=$rc_s lunar=$rc_l"
[[ "$rc_s" -eq 0 && "$rc_l" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+v4-n80 on 4,5/:8003"
log "DONE"
