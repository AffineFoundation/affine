#!/usr/bin/env bash
# p3811: R752 REFUTE → reap chall by pid → launch R765 TRAIN 4,5 + wait→merge + wait→n80.
# Keep R764 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3811-reap-r752] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R752 chall; keep /tmp/r752_merged; leave R764 TRAIN on 6,7"
stop_pidfile /root/logs/r752_sim_wvk7.pid "r752 sim"
stop_pidfile /root/logs/p3806_r752_lean.outer.pid "r752 lean outer"
stop_pidfile /root/logs/vllm_chall_r752.pid "r752 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r752 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r752_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r752 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_lunar_gpus45_p3806\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r752 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r752/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r752_refute_reaped.p3811
EXP=r765-marsplan-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr
log "launch R765 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus45_p3811.sh \
  >/root/logs/p3811_r765_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3811_r765_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r765_train_then_merge_p3811.sh \
  >/root/logs/p3811_r765_wait.nohup 2>&1 &
echo $! >/root/logs/p3811_r765_wait.pid
nohup bash /root/mining_src/$EXP/wait_r765_merge_then_n80_p3811.sh \
  >/root/logs/p3811_r765_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3811_r765_wait_n80.pid
log "R765 lean=$(cat /root/logs/p3811_r765_lean_outer.pid) wait=$(cat /root/logs/p3811_r765_wait.pid) wait_n80=$(cat /root/logs/p3811_r765_wait_n80.pid)"
# confirm R764 still alive
if [[ -f /root/logs/r764_train.pid ]]; then
  p=$(cat /root/logs/r764_train.pid)
  if kill -0 "$p" 2>/dev/null; then log "R764 TRAIN still alive pid=$p OK"; else log "WARN R764 train not alive"; fi
fi
