#!/usr/bin/env bash
# p3714: R681 uplink free + CHALL loading → stage R679 MERGE_DONE brave→golden.
# Short HiRank HiBeta UltraExtra. Do NOT touch R681 :8003 / r681_merged.
# Chall waiter gated on R681 n80 clear. Never pkill -f. Never dual-pipe.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r679-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-ultraextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r679_p3714.log
: >"$LOG"
log() { echo "[p3714-r679] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r679-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3714.sh
WAIT_LOCAL=$EXP/wait_r679_scp_after_r681_then_chall_p3714.sh

log "verify no other brave tar uplink"
bt=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${bt:-99}" -eq 0 ]] || { log "FATAL brave already tar-piping (count=$bt)"; exit 1; }

log "verify brave R679 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r679_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r679 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to golden; clear stale r679 dest ONLY (keep r681)"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_golden_gpus45_p3714.sh && bash -n $REMOTE_DIR/wait_r679_scp_after_r681_then_chall_p3714.sh && echo SYNTAX_OK"

ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
# kill hung r679 tar xf only by PID scan (never pkill -f); do not touch r681
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r679_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3714] kill hung r679 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
rm -rf /tmp/r679_merged
rm -f /root/logs/r679_scp_ready.done /root/logs/r679_chall_n80_launched.p3714
mkdir -p /tmp/r679_merged
# free ~200G REFUTE merges if needed (keep r681/r637/r647)
for d in /tmp/r609_merged /tmp/r616_merged /tmp/r622_merged /tmp/r627_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "freed $d"
done
nohup bash /root/mining_src/r679-chall/wait_r679_scp_after_r681_then_chall_p3714.sh \
  >/root/logs/p3714_r679_wait.nohup 2>&1 &
echo $! >/root/logs/p3714_r679_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3714_r679_wait.pid)"
# confirm R681 chall still alive (do not disturb)
if [[ -f /root/logs/vllm_chall_r681.pid ]]; then
  p=$(cat /root/logs/vllm_chall_r681.pid)
  kill -0 "$p" 2>/dev/null && echo "R681_CHALL_ALIVE pid=$p" || echo "R681_CHALL_GONE"
fi
REMOTE

log "begin host-relay tar pipe brave→golden (~67G R679) solo (R681 chall untouched; R675 R252→lunar untouched)"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r679_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r679_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r679_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r679_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r679_merged"
rc_b=${PIPESTATUS[0]:-0}
rc_g=${PIPESTATUS[1]:-0}
set -e
log "pipe exit brave=$rc_b golden=$rc_g"
[[ "$rc_b" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter gated on R681 clear then chall+v4-n80 on 4,5/:8003"
log "DONE"
