#!/usr/bin/env bash
# p3822: R757 REFUTE → reap chall by pid → launch R772 TRAIN 4,5 + wait→merge + wait→n80.
# Keep R766 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3822-reap-r757] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R757 chall; keep /tmp/r757_merged; leave R766 TRAIN on 6,7"
stop_pidfile /root/logs/r757_sim_wvk7.pid "r757 sim"
stop_pidfile /root/logs/p3804_r757_lean.outer.pid "r757 lean outer"
stop_pidfile /root/logs/vllm_chall_r757.pid "r757 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r757 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r757_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r757 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_zesty_gpus45_p3804\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r757 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r757/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r757_refute_reaped.p3822
EXP=r772-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-lolr
log "launch R772 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus45_p3822.sh   >/root/logs/p3822_r772_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3822_r772_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r772_train_then_merge_p3822.sh   >/root/logs/p3822_r772_wait.nohup 2>&1 &
echo $! >/root/logs/p3822_r772_wait.pid
nohup bash /root/mining_src/$EXP/wait_r772_merge_then_n80_p3822.sh   >/root/logs/p3822_r772_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3822_r772_wait_n80.pid
log "R772 lean=$(cat /root/logs/p3822_r772_lean_outer.pid) wait=$(cat /root/logs/p3822_r772_wait.pid) wait_n80=$(cat /root/logs/p3822_r772_wait_n80.pid)"
