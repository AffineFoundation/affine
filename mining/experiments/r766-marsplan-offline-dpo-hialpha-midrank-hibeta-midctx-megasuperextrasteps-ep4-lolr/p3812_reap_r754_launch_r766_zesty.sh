#!/usr/bin/env bash
# p3812: R754 REFUTE → reap chall by pid → launch R766 TRAIN 6,7 + wait→merge + wait→n80.
# Keep R757 TRAIN 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3812-reap-r754] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R754 chall; keep /tmp/r754_merged; leave R757 TRAIN on 4,5"
stop_pidfile /root/logs/r754_sim_wvk7.pid "r754 sim"
stop_pidfile /root/logs/p3799_r754_lean.outer.pid "r754 lean outer"
stop_pidfile /root/logs/vllm_chall_r754.pid "r754 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r754 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r754_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r754 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_zesty_gpus67_p3799\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r754 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r754/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r754_refute_reaped.p3812
EXP=r766-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-lolr
log "launch R766 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3812.sh \
  >/root/logs/p3812_r766_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3812_r766_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r766_train_then_merge_p3812.sh \
  >/root/logs/p3812_r766_wait.nohup 2>&1 &
echo $! >/root/logs/p3812_r766_wait.pid
nohup bash /root/mining_src/$EXP/wait_r766_merge_then_n80_p3812.sh \
  >/root/logs/p3812_r766_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3812_r766_wait_n80.pid
log "R766 lean=$(cat /root/logs/p3812_r766_lean_outer.pid) wait=$(cat /root/logs/p3812_r766_wait.pid) wait_n80=$(cat /root/logs/p3812_r766_wait_n80.pid)"
if [[ -f /root/logs/r757_train.pid ]]; then
  p=$(cat /root/logs/r757_train.pid)
  if kill -0 "$p" 2>/dev/null; then log "R757 TRAIN still alive pid=$p OK"; else log "WARN R757 train not alive"; fi
fi
