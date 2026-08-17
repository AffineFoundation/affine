#!/usr/bin/env bash
# p3666: R651 MERGE_DONE → arm Soft MidRank HiBeta ep3×LoLR → lunar 4,5/:8003.
# Gate brave uplink: wait R634 SCP_READY, then wait R647 SCP_READY (R647 already
# first in queue) — never dual-pipe brave. Warm TK on lunar; leave R537 :8002.
# Never pkill -f.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r651-r252-offline-dpo-hialpha-midrank-hibeta-softctx-megaextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR" "$ROOT/.ralph" "$EXP"
LOG=$LOGDIR/arm_r651_lunar_p3666.log
: >"$LOG"
log() { echo "[p3666-arm-r651] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes
  -o ServerAliveInterval=20 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
LUNAR_HOST=150.136.46.118
LUNAR_PORT=20299
ZESTY_HOST=86.38.182.95
ZESTY_PORT=20299
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r651-chall

log "verify brave /tmp/r651_merged complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r651_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r651 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "ensure lunar 4,5 free; leave R537 :8002 + TK 0-3"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
reap() {
  local pid=$1 why=$2
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  echo "kill $pid ($why)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for f in /root/logs/p3662_r643_lean_outer.pid /root/logs/p3651_r643_lean_outer.pid \
  /root/logs/r643_sim_reign34.pid /root/logs/vllm_chall_r643.pid \
  /root/logs/p3666_r651_lean_outer.pid /root/logs/vllm_chall_r651.pid; do
  if [[ -f "$f" ]]; then
    reap "$(cat "$f")" "stale $f"
    rm -f "$f"
  fi
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid" ":8003 listener"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid" "r643/r651_merged argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r(643|651)_merged/ && !/awk/ {print $1}')
rm -f /root/logs/r643_chall_n80_launched.p3651 /root/logs/r651_chall_n80_launched.p3666 \
  /root/logs/r651_scp_ready.done
for i in $(seq 1 40); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 5000 ]]; then
    echo "GPUs 4,5 free used_mib=$used"
    break
  fi
  sleep 2
done
nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used --format=csv
curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null && echo T_OK || echo T_MISSING
curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null && echo K_OK || echo K_MISSING
curl -sf -m 3 http://127.0.0.1:8002/v1/models >/dev/null && echo R537_OK || echo R537_MISSING
REMOTE

log "upload wait+lean to lunar"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$EXP/wait_r651_scp_king_then_chall_p3666.sh" \
  "$EXP/lean_chall_n80_lunar_gpus45_p3666.sh" \
  "root@$LUNAR_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_lunar_gpus45_p3666.sh && bash -n $REMOTE_DIR/wait_r651_scp_king_then_chall_p3666.sh && echo SYNTAX_OK"

log "start waiter on lunar"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
nohup bash /root/mining_src/r651-chall/wait_r651_scp_king_then_chall_p3666.sh \
  >/root/logs/p3666_r651_wait.nohup 2>&1 &
echo $! >/root/logs/p3666_r651_wait.pid
echo "wait_pid=$(cat /root/logs/p3666_r651_wait.pid)"
REMOTE

log "write gated host-relay (R634 then R647 free brave) + start nohup"
cat >"$ROOT/.ralph/p3666_r651_host_relay.sh" <<'RELAY'
#!/usr/bin/env bash
set -euo pipefail
LOG=/home/const/subnet120/mining/.ralph/p3666_r651_host_relay.log
log() { echo "[p3666-relay] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
LUNAR_HOST=150.136.46.118; LUNAR_PORT=20299
ZESTY_HOST=86.38.182.95; ZESTY_PORT=20299
GOLD_HOST=38.127.229.127; GOLD_PORT=40299
: >"$LOG"
log "gate1: wait R634 SCP to free brave uplink"
for i in $(seq 1 720); do
  ready=0
  if ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
      "test -f /root/logs/r634_scp_ready.done && echo READY" 2>/dev/null | grep -q READY; then
    ready=1
  fi
  r634_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "ps aux | grep 'tar cf - r634_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
  if [[ "$ready" -eq 1 ]] || [[ "${r634_tar:-99}" -eq 0 ]]; then
    log "R634 uplink free (ready=$ready r634_tar=$r634_tar) after ${i} polls"
    break
  fi
  if (( i % 6 == 0 )); then
    sz=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
      "du -sm /tmp/r634_merged 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
    log "still waiting R634… poll=$i size_mib=$sz r634_tar=$r634_tar"
  fi
  if [[ "$i" -eq 720 ]]; then
    log "TIMEOUT waiting R634 uplink"
    exit 1
  fi
  sleep 30
done

log "gate2: wait R647 SCP (first in queue) to finish before R651 pipe"
for i in $(seq 1 720); do
  ready=0
  if ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "test -f /root/logs/r647_scp_ready.done && echo READY" 2>/dev/null | grep -q READY; then
    ready=1
  fi
  r647_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "ps aux | grep 'tar cf - r647_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
  # Also treat "R647 relay not running and no tar" as free only after R634 done —
  # but prefer explicit r647_scp_ready (R647 arm already launched).
  if [[ "$ready" -eq 1 ]]; then
    log "R647 SCP_READY after ${i} polls"
    break
  fi
  if (( i % 6 == 0 )); then
    log "still waiting R647… poll=$i ready=$ready r647_tar=$r647_tar"
  fi
  if [[ "$i" -eq 720 ]]; then
    log "TIMEOUT waiting R647 SCP"
    exit 1
  fi
  sleep 30
done

# brief settle so r647 tar fully exits
sleep 5
r647_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep 'tar cf - r647_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
if [[ "${r647_tar:-99}" -ne 0 ]]; then
  log "wait extra for r647 tar to exit…"
  for j in $(seq 1 60); do
    r647_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "ps aux | grep 'tar cf - r647_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
    [[ "${r647_tar:-99}" -eq 0 ]] && break
    sleep 10
  done
fi
r647_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep 'tar cf - r647_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
if [[ "${r647_tar:-99}" -ne 0 ]]; then
  log "FATAL r647 tar still on brave after gate"
  exit 1
fi

log "prep lunar dest"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "rm -rf /tmp/r651_merged"
log "begin host-relay tar pipe brave→lunar (~67G R651) solo"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r651_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r651_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r651_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r651_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r651_merged"
rc_b=${PIPESTATUS[0]}
rc_l=${PIPESTATUS[1]}
set -e
log "pipe exit brave=$rc_b lunar=$rc_l"
if [[ "$rc_b" -ne 0 || "$rc_l" -ne 0 ]]; then
  log "FATAL pipe failed"
  exit 1
fi
log "SCP_READY confirmed"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "cat /root/logs/r651_scp_ready.done; ls /tmp/r651_merged/model-*-of-*.safetensors | wc -l; du -sh /tmp/r651_merged"
log "DONE"
RELAY
chmod +x "$ROOT/.ralph/p3666_r651_host_relay.sh"
nohup bash "$ROOT/.ralph/p3666_r651_host_relay.sh" \
  >"$ROOT/.ralph/p3666_r651_host_relay.nohup" 2>&1 &
echo $! >"$ROOT/.ralph/p3666_r651_host_relay.pid"
log "host_relay pid=$(cat $ROOT/.ralph/p3666_r651_host_relay.pid) log=$ROOT/.ralph/p3666_r651_host_relay.log"
log "ARMED R651 → lunar 4,5/:8003 (after R634 then R647)"
