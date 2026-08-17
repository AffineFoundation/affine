#!/usr/bin/env bash
# p3723: queue R683 MERGE_DONE brave→golden AFTER R680 uplink clears (no dual-pipe).
# Short MidRank HiBeta UltraExtra. Chall waiter gated on R680 n80 clear.
# Never pkill -f. Leave R675 R252→lunar alone. Leave R637 :8004 alone.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r683-r252-offline-dpo-hialpha-midrank-hibeta-shortctx-ultraextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r683_p3723.log
: >"$LOG"
log() { echo "[p3723-r683] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r683-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3723.sh
WAIT_LOCAL=$EXP/wait_r683_scp_after_r680_then_chall_p3723.sh

log "GATE: wait until R680 zesty→golden tar uplink is clear (no dual-pipe)"
for i in $(seq 1 720); do
  zesty_tar=$(pgrep -c -f 'tar cf - r680_merged' 2>/dev/null || true)
  # also detect via ssh on zesty if host process naming differs
  gold_xf=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
  relay=$(pgrep -c -f 'host_relay_r680_zesty_to_golden_p3719' 2>/dev/null || true)
  [[ $((i % 3)) -eq 1 ]] && log "gate poll i=$i zesty_tar=$zesty_tar gold_xf=$gold_xf relay=$relay"
  if [[ "${zesty_tar:-0}" -eq 0 && "${gold_xf:-0}" -eq 0 && "${relay:-0}" -eq 0 ]]; then
    log "R680 uplink clear — proceed with R683 SCP"
    break
  fi
  if [[ "$i" -eq 720 ]]; then
    log "FATAL timeout waiting for R680 uplink clear"
    exit 1
  fi
  sleep 20
done

log "verify no other brave tar uplink"
bt=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${bt:-99}" -eq 0 ]] || { log "FATAL brave already tar-piping (count=$bt)"; exit 1; }

log "verify brave R683 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r683_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r683 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to golden; clear stale r683 dest ONLY (keep r680)"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_golden_gpus45_p3723.sh && bash -n $REMOTE_DIR/wait_r683_scp_after_r680_then_chall_p3723.sh && echo SYNTAX_OK"

ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
# kill hung r683 tar xf only by PID scan (never pkill -f); do not touch r680
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r683_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3723] kill hung r683 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
rm -rf /tmp/r683_merged
rm -f /root/logs/r683_scp_ready.done /root/logs/r683_chall_n80_launched.p3723
mkdir -p /tmp/r683_merged
# free old REFUTE merges if needed (keep r680/r637)
for d in /tmp/r679_merged /tmp/r681_merged /tmp/r663_merged /tmp/r653_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "freed $d"
done
nohup bash /root/mining_src/r683-chall/wait_r683_scp_after_r680_then_chall_p3723.sh \
  >/root/logs/p3723_r683_wait.nohup 2>&1 &
echo $! >/root/logs/p3723_r683_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3723_r683_wait.pid)"
# confirm R680 chall/scp state (do not disturb)
if [[ -f /root/logs/vllm_chall_r680.pid ]]; then
  p=$(cat /root/logs/vllm_chall_r680.pid)
  kill -0 "$p" 2>/dev/null && echo "R680_CHALL_ALIVE pid=$p" || echo "R680_CHALL_GONE"
fi
ls /tmp/r680_merged/model-*-of-*.safetensors 2>/dev/null | wc -l | awk '{print "r680_shards="$1}'
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe brave→golden (~66G R683) solo after R680 uplink clear"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r683_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r683_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r683_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r683_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r683_merged"
rc_b=${PIPESTATUS[0]:-0}
rc_g=${PIPESTATUS[1]:-0}
set -e
log "pipe exit brave=$rc_b golden=$rc_g"
[[ "$rc_b" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter gated on R680 clear then chall+v4-n80 on 4,5/:8003"
log "DONE"
