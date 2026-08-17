#!/usr/bin/env bash
# p3699: after R653 v4 REFUTE, arm R663 MERGE_DONE crown→golden 4,5/:8003.
# Long HiRank LoBeta LongCtx Mega ep3×LoLR (amplify R648 ~0.68×).
# Never dual-pipe R252 (R655 R252→lunar untouched). Never pkill -f. Leave R637 :8004.
set -euo pipefail

ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r663-r252-offline-dpo-hialpha-hirank-lobeta-longctx-megaextrasteps-ep3-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r663_p3699.log
: >"$LOG"
log() { echo "[p3699-r663] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
CROWN_HOST=95.133.253.90
CROWN_PORT=40099
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
REMOTE_DIR=/root/mining_src/r663-chall
LEAN_LOCAL=$EXP/lean_chall_n80_golden_gpus45_p3699.sh
WAIT_LOCAL=$EXP/wait_r663_scp_king_then_chall_p3699.sh

log "verify no other crown tar uplink"
ct=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "ps aux | grep -E 'tar cf - r[0-9]+_merged' | grep -v grep | wc -l" 2>/dev/null || echo 99)
[[ "${ct:-99}" -eq 0 ]] || { log "FATAL crown already tar-piping (count=$ct)"; exit 1; }

log "verify crown R663 source complete"
n_src=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "ls /tmp/r663_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "crown r663 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

log "upload lean+wait to golden; reap R653 chall 4,5; clear dest"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "mkdir -p $REMOTE_DIR /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$GOLD_PORT" "$LEAN_LOCAL" "$WAIT_LOCAL" "root@$GOLD_HOST:$REMOTE_DIR/"
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
  "chmod +x $REMOTE_DIR/*.sh; bash -n $REMOTE_DIR/lean_chall_n80_golden_gpus45_p3699.sh && bash -n $REMOTE_DIR/wait_r663_scp_king_then_chall_p3699.sh && echo SYNTAX_OK"

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
for f in /root/logs/vllm_chall_r653.pid /root/logs/r653_sim_wvk7.pid \
  /root/logs/p3693_r653_lean_outer.pid /root/logs/p3693_r653_wait.pid \
  /root/logs/vllm_chall_r663.pid /root/logs/p3699_r663_wait.pid; do
  [[ -f "$f" ]] && reap "$(cat "$f" 2>/dev/null || true)"
done
# lean outer still holding r653
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_golden_gpus45_p3693|wait_r653_scp/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r653_merged/ && !/awk/ {print $1}')
rm -rf /tmp/r653_merged /tmp/r663_merged
rm -f /root/logs/r653_scp_ready.done /root/logs/r653_chall_n80_launched.p3693 \
  /root/logs/r663_scp_ready.done /root/logs/r663_chall_n80_launched.p3699
mkdir -p /tmp/r663_merged
nohup bash /root/mining_src/r663-chall/wait_r663_scp_king_then_chall_p3699.sh \
  >/root/logs/p3699_r663_wait.nohup 2>&1 &
echo $! >/root/logs/p3699_r663_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3699_r663_wait.pid)"
for i in $(seq 1 40); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 5000 ]]; then
    echo "GPUs 4,5 free used_mib=$used"
    break
  fi
  sleep 2
done
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used --format=csv
code=$(curl -s -o /tmp/king_models_p3699_pre.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3699_pre.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
echo "[p3699-pre] king code=$code id=$id"
echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'
curl -sf -m 3 http://127.0.0.1:8004/v1/models >/dev/null && echo R637_8004_OK || echo R637_8004_MISSING
df -h / | tail -1
REMOTE

log "begin host-relay tar pipe crown→golden (~66G R663) solo (R655 R252→lunar untouched)"
set +e
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "cd /tmp && tar cf - r663_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r663_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r663_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r663_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r663_merged"
rc_c=${PIPESTATUS[0]}
rc_g=${PIPESTATUS[1]}
set -e
log "pipe exit crown=$rc_c golden=$rc_g"
[[ "$rc_c" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL pipe failed"; exit 1; }
log "SCP_READY confirmed; waiter handles chall+v4-n80 on 4,5/:8003"
log "DONE"
