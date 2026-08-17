#!/usr/bin/env bash
# p3661: R633 after R634 pipeline.done — ALSO wait for R647 brave→golden pipe
# to finish (no `tar cf - r647_merged` on brave) so we never dual-pipe.
# Never pkill -f. Leave R634 6,7 alone until its own reap.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r633-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megaextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r633_p3661.log
: >"$LOG"
log() { echo "[p3661-r633] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=20 -o ServerAliveCountMax=12 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
ZESTY_HOST=86.38.182.95
ZESTY_PORT=20299
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r633-chall
LEAN_LOCAL=$EXP/lean_chall_n80_zesty_gpus45_p3623.sh
WAIT_LOCAL=$EXP/wait_r633_scp_king_then_chall_p3623.sh

log "armed: wait R634 pipeline.done + R647 uplink clear, then solo R633 (timeout 10h)"
for i in $(seq 1 3600); do
  ready=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
    'if [[ -f /root/logs/r634_reign34_pipeline.done ]]; then
       echo "ok"
     elif [[ -f /root/logs/r634_scp_ready.done ]]; then
       echo "scp_only"
     else
       n=$(ls /tmp/r634_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0)
       echo "wait:$n"
     fi' 2>/dev/null || echo "sshfail")
  if [[ "$ready" == ok ]]; then
    log "R634 pipeline.done — next check R647 uplink (i=$i)"
    break
  fi
  [[ $((i % 6)) -eq 0 ]] && log "poll i=$i r634=$ready"
  if [[ "$i" -eq 3600 ]]; then
    log "TIMEOUT waiting R634 pipeline.done"
    exit 1
  fi
  sleep 10
done

log "wait R647 SCP_READY on golden (p3661 armed ahead of R633; never dual-pipe)"
for j in $(seq 1 720); do
  if ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "test -f /root/logs/r647_scp_ready.done && echo READY" 2>/dev/null | grep -q READY; then
    r647_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "ps aux | grep 'tar cf - r647_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
    if [[ "${r647_tar:-99}" -eq 0 ]]; then
      log "R647 SCP_READY + no tar after ${j} polls"
      break
    fi
    log "R647 SCP_READY but tar still draining (tar=$r647_tar) poll=$j"
  fi
  if (( j % 6 == 0 )); then
    sz=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "du -sm /tmp/r647_merged 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
    log "still waiting R647 SCP… poll=$j size_mib=$sz"
  fi
  if [[ "$j" -eq 720 ]]; then
    log "TIMEOUT waiting R647 SCP_READY"
    exit 1
  fi
  sleep 30
done

log "verify brave R633 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r633_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r633 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "refresh lean+wait on zesty; purge REFUTE leftovers for disk"
ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "mkdir -p $REMOTE_DIR /root/logs"
scp "${SSH_OPTS[@]}" -P "$ZESTY_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$ZESTY_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "chmod +x $REMOTE_DIR/*.sh"

ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'r633_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3661] kill hung r633 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
for d in /tmp/r629_merged /tmp/r636_merged; do
  if [[ -d "$d" ]]; then
    echo "[p3661] purge $d (REFUTE leftover)"
    rm -rf "$d"
  fi
done
if [[ -f /root/logs/vllm_chall_r634.pid ]]; then
  pid=$(cat /root/logs/vllm_chall_r634.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[p3661] stop R634 chall pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f /root/logs/vllm_chall_r634.pid
fi
rm -rf /tmp/r633_merged
rm -f /root/logs/r633_scp_ready.done /root/logs/r633_chall_n80_launched.p3623
mkdir -p /tmp/r633_merged
if [[ -f /root/logs/p3623_r633_wait.pid ]] && kill -0 "$(cat /root/logs/p3623_r633_wait.pid)" 2>/dev/null; then
  echo "waiter_alive pid=$(cat /root/logs/p3623_r633_wait.pid)"
else
  nohup bash /root/mining_src/r633-chall/wait_r633_scp_king_then_chall_p3623.sh \
    >/root/logs/p3623_r633_wait.nohup 2>&1 &
  echo $! >/root/logs/p3623_r633_wait.pid
  echo "waiter_started pid=$(cat /root/logs/p3623_r633_wait.pid)"
fi
code=$(curl -s -o /tmp/king_models_p3661_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3661_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3661-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
[[ -f /root/logs/swap_king_reign34_zesty_p3596.done ]] || date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/swap_king_reign34_zesty_p3596.done
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe brave→zesty (~66G R633) solo after R647"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r633_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r633_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r633_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r633_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r633_merged"
rc_b=${PIPESTATUS[0]}
rc_z=${PIPESTATUS[1]}
set -e
log "pipe exit brave=$rc_b zesty=$rc_z"
[[ "$rc_b" -eq 0 && "$rc_z" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+n80 on 4,5/:8002"
log "DONE"
