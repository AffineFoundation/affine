#!/usr/bin/env bash
# p3800: R745 REFUTE → reap chall by pid → launch R755 TRAIN 4,5 + wait→merge + wait→n80. Keep R746 N80 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3800-reap-r745] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R745 chall; keep /tmp/r745_merged; leave R746 N80 on 6,7"
stop_pidfile /root/logs/r745_sim_wvk7.pid "r745 sim"
stop_pidfile /root/logs/p3798_r745_lean.outer.pid "r745 lean outer"
stop_pidfile /root/logs/vllm_chall_r745.pid "r745 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r745 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r745_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r745 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_golden_gpus45_p3798\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r745_refute_reaped.p3800
EXP=r755-r252-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-lolr
log "launch R755 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_golden_gpus45_p3800.sh \
  >/root/logs/p3800_r755_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3800_r755_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r755_train_then_merge_p3800.sh \
  >/root/logs/p3800_r755_wait.nohup 2>&1 &
echo $! >/root/logs/p3800_r755_wait.pid
nohup bash /root/mining_src/$EXP/wait_r755_merge_then_n80_p3800.sh \
  >/root/logs/p3800_r755_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3800_r755_wait_n80.pid
log "R755 lean=$(cat /root/logs/p3800_r755_lean_outer.pid) wait=$(cat /root/logs/p3800_r755_wait.pid) wait_n80=$(cat /root/logs/p3800_r755_wait_n80.pid)"
