#!/usr/bin/env bash
# p3719: R680 MERGE_DONE zesty→golden 4,5/:8003 (Long MidRank HiBeta UltraExtra).
# Independent of R675 (R252→lunar). Never dual-pipe golden recv. Never pkill -f.
# Leave R637 :8004 alone.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r680-r252-offline-dpo-hialpha-midrank-hibeta-longctx-ultraextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r680_p3719.log
: >"$LOG"
log() { echo "[p3719-r680] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
ZESTY_HOST=86.38.182.95
ZESTY_PORT=20299
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r680-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3719.sh
WAIT_LOCAL=$EXP/wait_r680_scp_king_then_chall_p3719.sh

log "verify no other zesty tar uplink"
zt=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${zt:-99}" -eq 0 ]] || { log "FATAL zesty already tar-piping (count=$zt)"; exit 1; }

log "verify golden has no live tar xf (R675 is lunar — golden should be idle)"
gt=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
  'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
[[ "${gt:-0}" -eq 0 ]] || { log "FATAL golden already tar-receiving (count=$gt)"; exit 1; }

log "verify zesty R680 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
  "ls /tmp/r680_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "zesty r680 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to golden; clear stale r680 dest; arm waiter"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_golden_gpus45_p3719.sh && bash -n $REMOTE_DIR/wait_r680_scp_king_then_chall_p3719.sh && echo SYNTAX_OK"

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
for f in /root/logs/vllm_chall_r679.pid /root/logs/vllm_chall_r681.pid \
  /root/logs/vllm_chall_r680.pid \
  /root/logs/p3714_r679_wait.pid /root/logs/p3714_r679_lean_outer.pid \
  /root/logs/p3711_r681_wait.pid /root/logs/p3711_r681_lean_outer.pid \
  /root/logs/p3719_r680_wait.pid /root/logs/r679_sim_wvk7.pid \
  /root/logs/r681_sim_wvk7.pid; do
  [[ -f "$f" ]] && reap "$(cat "$f" 2>/dev/null || true)"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
# kill hung r679/r681/r680 tar xf only by PID scan (never pkill -f)
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r(679|681|680)_merged' && echo "$cmd" | grep -q 'tar xf'; then
    echo "[p3719] kill hung tar xf pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
done
rm -rf /tmp/r680_merged
rm -f /root/logs/r680_scp_ready.done /root/logs/r680_chall_n80_launched.p3719
mkdir -p /tmp/r680_merged
nohup bash /root/mining_src/r680-chall/wait_r680_scp_king_then_chall_p3719.sh \
  >/root/logs/p3719_r680_wait.nohup 2>&1 &
echo $! >/root/logs/p3719_r680_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3719_r680_wait.pid)"
code=$(curl -s -o /tmp/king_models_p3719_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3719_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3719-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
curl -sf -m 3 http://127.0.0.1:8004/v1/models >/dev/null && echo R637_8004_OK || echo R637_8004_MISSING
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe zesty→golden (~66G R680) solo (R675 R252→lunar untouched)"
set +e
ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "cd /tmp && tar cf - r680_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r680_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r680_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r680_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r680_merged"
rc_z=${PIPESTATUS[0]:-0}
rc_g=${PIPESTATUS[1]:-0}
set -e
log "pipe exit zesty=$rc_z golden=$rc_g"
[[ "$rc_z" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+v4-n80 on 4,5/:8003"
log "DONE"
