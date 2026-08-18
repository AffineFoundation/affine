#!/usr/bin/env bash
# p3829: R765 REFUTE → reap chall :8002 on 4,5 → R779 Soft HiRank LoBeta SoftCtx Mega TRAIN.
# Keep R778 TRAIN on 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3829-reap-r765] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R765 chall/n80; keep R778 on 6,7; keep /tmp/r765_merged"
stop_pidfile /root/logs/vllm_chall_r765.pid "r765 vllm"
stop_pidfile /root/logs/r765_sim_wvk7.pid "r765 sim"
stop_pidfile /root/logs/p3811_r765_lean.outer.pid "r765 lean outer"
stop_pidfile /root/logs/p3811_r765_wait_n80.pid "r765 wait_n80"
stop_pidfile /root/logs/p3811_r765_wait.pid "r765 wait"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r765 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r765_merged|run_sim_duel.py.*r765|lean_chall_n80_lunar_gpus45_p3811/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r765_refute_reaped.p3829
EXP=r779-marsplan-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-lolr
log "launch R779 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus45_p3829.sh   >/root/logs/p3829_r779_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3829_r779_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r779_train_then_merge_p3829.sh   >/root/logs/p3829_r779_wait.nohup 2>&1 &
echo $! >/root/logs/p3829_r779_wait.pid
nohup bash /root/mining_src/$EXP/wait_r779_merge_then_n80_p3829.sh   >/root/logs/p3829_r779_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3829_r779_wait_n80.pid
log "R779 lean=$(cat /root/logs/p3829_r779_lean_outer.pid) wait=$(cat /root/logs/p3829_r779_wait.pid) wait_n80=$(cat /root/logs/p3829_r779_wait_n80.pid)"
