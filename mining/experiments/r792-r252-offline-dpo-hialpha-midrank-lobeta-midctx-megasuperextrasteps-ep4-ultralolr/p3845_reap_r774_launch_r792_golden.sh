#!/usr/bin/env bash
# p3845: R774 REFUTE → reap chall :8002 on 4,5 → R792 MidCtx MidRank LoBeta Mega UltraLoLR TRAIN.
# Sibling R791 occupies 6,7. Never pkill -f. Keep /tmp/r774_merged.
set -euo pipefail
log() { echo "[p3845-reap-r774] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R774 chall/n80 on 4,5; keep /tmp/r774_merged"
stop_pidfile /root/logs/vllm_chall_r774.pid "r774 vllm"
stop_pidfile /root/logs/r774_sim_wvk7.pid "r774 sim"
stop_pidfile /root/logs/p3822_r774_lean.outer.pid "r774 lean outer"
stop_pidfile /root/logs/p3844_r774_lean.outer.pid "r774 lean outer p3844"
stop_pidfile /root/logs/p3822_r774_wait_n80.pid "r774 wait_n80"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r774 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r774_merged|run_sim_duel.py.*r774|lean_chall_n80_golden_gpus45/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r774_refute_reaped.p3845
EXP=r792-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R792 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_golden_gpus45_p3845.sh   >/root/logs/p3845_r792_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r792_train_then_merge_p3845.sh   >/root/logs/p3845_r792_wait.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_wait.pid
nohup bash /root/mining_src/$EXP/wait_r792_merge_then_n80_p3845.sh   >/root/logs/p3845_r792_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_wait_n80.pid
log "R792 lean=$(cat /root/logs/p3845_r792_lean_outer.pid) wait=$(cat /root/logs/p3845_r792_wait.pid) wait_n80=$(cat /root/logs/p3845_r792_wait_n80.pid)"
