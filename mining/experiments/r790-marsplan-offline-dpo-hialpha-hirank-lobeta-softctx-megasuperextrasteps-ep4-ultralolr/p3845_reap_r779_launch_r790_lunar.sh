#!/usr/bin/env bash
# p3845: R779 REFUTE → reap chall :8002 on 4,5 → R790 Soft HiRank LoBeta SoftCtx Mega UltraLoLR TRAIN.
# Keep R789 TRAIN on 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3845-reap-r779] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R779 chall/n80; keep R789 on 6,7; keep /tmp/r779_merged"
stop_pidfile /root/logs/vllm_chall_r779.pid "r779 vllm"
stop_pidfile /root/logs/r779_sim_wvk7.pid "r779 sim"
stop_pidfile /root/logs/p3829_r779_lean.outer.pid "r779 lean outer"
stop_pidfile /root/logs/p3829_r779_wait_n80.pid "r779 wait_n80"
stop_pidfile /root/logs/p3829_r779_wait.pid "r779 wait"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r779 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r779_merged|run_sim_duel.py.*r779|lean_chall_n80_lunar_gpus45_p3829/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r779_refute_reaped.p3845
EXP=r790-marsplan-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R790 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus45_p3845.sh   >/root/logs/p3845_r790_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3845_r790_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r790_train_then_merge_p3845.sh   >/root/logs/p3845_r790_wait.nohup 2>&1 &
echo $! >/root/logs/p3845_r790_wait.pid
nohup bash /root/mining_src/$EXP/wait_r790_merge_then_n80_p3845.sh   >/root/logs/p3845_r790_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3845_r790_wait_n80.pid
log "R790 lean=$(cat /root/logs/p3845_r790_lean_outer.pid) wait=$(cat /root/logs/p3845_r790_wait.pid) wait_n80=$(cat /root/logs/p3845_r790_wait_n80.pid)"
