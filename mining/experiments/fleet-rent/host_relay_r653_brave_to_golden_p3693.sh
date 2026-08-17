#!/usr/bin/env bash
# p3693: after R633 v4 REFUTE, arm R653 MERGE_DONE brave→golden 4,5/:8003.
# Short MidRank LoBeta ShortCtx Mega ep3×LoLR. Never dual-pipe (R655 is R252→lunar).
# Never pkill -f. Leave R637 :8004.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r653-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-megaextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r653_p3693.log
: >"$LOG"
log() { echo "[p3693-r653] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r653-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3693.sh
WAIT_LOCAL=$EXP/wait_r653_scp_king_then_chall_p3693.sh

log "verify no other brave tar uplink"
bt=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${bt:-99}" -eq 0 ]] || { log "FATAL brave already tar-piping (count=$bt)"; exit 1; }

log "verify brave R653 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r653_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r653 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to golden; clear stale r653 dest; reap free 4,5"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_golden_gpus45_p3693.sh && bash -n $REMOTE_DIR/wait_r653_scp_king_then_chall_p3693.sh && echo SYNTAX_OK"

ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  echo "kill $pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for f in /root/logs/vllm_chall_r633.pid /root/logs/vllm_chall_r652.pid /root/logs/vllm_chall_r653.pid /root/logs/p3693_r653_wait.pid; do
  [[ -f "$f" ]] && reap "$(cat "$f" 2>/dev/null || true)"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
# kill hung r653 tar xf only by PID scan (never pkill -f)
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'r653_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3693] kill hung r653 tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
rm -rf /tmp/r652_merged /tmp/r653_merged
rm -f /root/logs/r653_scp_ready.done /root/logs/r653_chall_n80_launched.p3693
mkdir -p /tmp/r653_merged
nohup bash /root/mining_src/r653-chall/wait_r653_scp_king_then_chall_p3693.sh \
  >/root/logs/p3693_r653_wait.nohup 2>&1 &
echo $! >/root/logs/p3693_r653_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3693_r653_wait.pid)"
code=$(curl -s -o /tmp/king_models_p3693_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3693_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3693-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
curl -sf -m 3 http://127.0.0.1:8004/v1/models >/dev/null && echo R637_8004_OK || echo R637_8004_MISSING
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe brave→golden (~66G R653) solo (R655 R252→lunar untouched)"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r653_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r653_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r653_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r653_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r653_merged"
rc_b=${PIPESTATUS[0]}
rc_g=${PIPESTATUS[1]}
set -e
log "pipe exit brave=$rc_b golden=$rc_g"
[[ "$rc_b" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+v4-n80 on 4,5/:8003"
log "DONE"
