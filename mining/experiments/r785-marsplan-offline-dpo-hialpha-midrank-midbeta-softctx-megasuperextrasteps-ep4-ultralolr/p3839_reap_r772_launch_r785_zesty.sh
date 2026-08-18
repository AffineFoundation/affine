#!/usr/bin/env bash
# p3839: R772 REFUTE → reap chall :8002 on 4,5 → R785 Soft MidRank MidBeta SoftCtx Mega UltraLoLR TRAIN.
# Keep R777 TRAIN on 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3839-reap-r772] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R772 chall/n80; keep R777 on 6,7; keep /tmp/r772_merged"
stop_pidfile /root/logs/vllm_chall_r772.pid "r772 vllm"
stop_pidfile /root/logs/r772_sim_wvk7.pid "r772 sim"
stop_pidfile /root/logs/p3822_r772_lean.outer.pid "r772 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r772 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r772_merged|run_sim_duel.py.*r772|lean_chall_n80_zesty_gpus45_p3822/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r772_refute_reaped.p3839
EXP=r785-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R785 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus45_p3839.sh   >/root/logs/p3839_r785_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3839_r785_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r785_train_then_merge_p3839.sh   >/root/logs/p3839_r785_wait.nohup 2>&1 &
echo $! >/root/logs/p3839_r785_wait.pid
nohup bash /root/mining_src/$EXP/wait_r785_merge_then_n80_p3839.sh   >/root/logs/p3839_r785_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3839_r785_wait_n80.pid
log "R785 lean=$(cat /root/logs/p3839_r785_lean_outer.pid) wait=$(cat /root/logs/p3839_r785_wait.pid) wait_n80=$(cat /root/logs/p3839_r785_wait_n80.pid)"
