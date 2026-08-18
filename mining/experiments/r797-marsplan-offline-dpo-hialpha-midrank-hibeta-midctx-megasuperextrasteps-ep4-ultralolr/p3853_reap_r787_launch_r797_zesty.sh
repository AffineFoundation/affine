#!/usr/bin/env bash
# p3853: R787 REFUTE → reap chall :8003 on 6,7 → R797 MidCtx MidRank HiBeta Mega UltraLoLR TRAIN.
# Keep R796 TRAIN on 4,5. Keep /tmp/r787_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3853-reap-r787] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R787 chall/n80; keep R796 on 4,5; keep /tmp/r787_merged"
stop_pidfile /root/logs/vllm_chall_r787.pid "r787 vllm"
stop_pidfile /root/logs/r787_sim_wvk7.pid "r787 sim"
stop_pidfile /root/logs/p3841_r787_lean.outer.pid "r787 lean outer"
stop_pidfile /root/logs/p3841_r787_lean_outer.pid "r787 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r787 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r787_merged|run_sim_duel.py.*r787|lean_chall_n80_zesty_gpus67_p3841/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r787_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r796' && continue
  stop_pid "$pid" "r787 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r787_refute_reaped.p3853
EXP=r797-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R797 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3853.sh   >/root/logs/p3853_r797_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3853_r797_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r797_train_then_merge_p3853.sh   >/root/logs/p3853_r797_wait.nohup 2>&1 &
echo $! >/root/logs/p3853_r797_wait.pid
nohup bash /root/mining_src/$EXP/wait_r797_merge_then_n80_p3853.sh   >/root/logs/p3853_r797_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3853_r797_wait_n80.pid
log "R797 lean=$(cat /root/logs/p3853_r797_lean_outer.pid) wait=$(cat /root/logs/p3853_r797_wait.pid) wait_n80=$(cat /root/logs/p3853_r797_wait_n80.pid)"
