#!/usr/bin/env bash
# p3827: R766 REFUTE → reap chall :8003 on 6,7 → R777 Soft MidRank HiBeta SoftCtx Mega TRAIN.
# Keep R772 TRAIN on 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3827-reap-r766] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1; local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid; pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"
  rm -f "$pidf"
}
log "START reap R766 chall/n80; keep R772 on 4,5; keep /tmp/r766_merged"
stop_pidfile /root/logs/vllm_chall_r766.pid "r766 vllm"
stop_pidfile /root/logs/r766_sim_wvk7.pid "r766 sim"
stop_pidfile /root/logs/p3812_r766_lean.outer.pid "r766 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r766 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r766_merged|run_sim_duel.py.*r766|lean_chall_n80_zesty_gpus67_p3812/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r766_refute_reaped.p3827
EXP=r777-marsplan-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-lolr
log "launch R777 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3827.sh   >/root/logs/p3827_r777_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3827_r777_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r777_train_then_merge_p3827.sh   >/root/logs/p3827_r777_wait.nohup 2>&1 &
echo $! >/root/logs/p3827_r777_wait.pid
nohup bash /root/mining_src/$EXP/wait_r777_merge_then_n80_p3827.sh   >/root/logs/p3827_r777_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3827_r777_wait_n80.pid
log "R777 lean=$(cat /root/logs/p3827_r777_lean_outer.pid) wait=$(cat /root/logs/p3827_r777_wait.pid) wait_n80=$(cat /root/logs/p3827_r777_wait_n80.pid)"
