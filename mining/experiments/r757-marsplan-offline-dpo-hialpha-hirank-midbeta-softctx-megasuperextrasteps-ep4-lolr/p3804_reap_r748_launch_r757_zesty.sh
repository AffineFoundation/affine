#!/usr/bin/env bash
# p3804: R748 REFUTE → reap chall by pid → launch R757 TRAIN 4,5 + wait→merge + wait→n80. Keep R754 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3804-reap-r748] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R748 chall; keep /tmp/r748_merged; leave R754 TRAIN on 6,7"
stop_pidfile /root/logs/r748_sim_wvk7.pid "r748 sim"
stop_pidfile /root/logs/p3802_r748_lean.outer.pid "r748 lean outer"
stop_pidfile /root/logs/vllm_chall_r748.pid "r748 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r748 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r748_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r748 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_zesty_gpus45_p3802\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r748_refute_reaped.p3804
EXP=r757-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-lolr
log "launch R757 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus45_p3804.sh \
  >/root/logs/p3804_r757_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3804_r757_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r757_train_then_merge_p3804.sh \
  >/root/logs/p3804_r757_wait.nohup 2>&1 &
echo $! >/root/logs/p3804_r757_wait.pid
nohup bash /root/mining_src/$EXP/wait_r757_merge_then_n80_p3804.sh \
  >/root/logs/p3804_r757_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3804_r757_wait_n80.pid
log "R757 lean=$(cat /root/logs/p3804_r757_lean_outer.pid) wait=$(cat /root/logs/p3804_r757_wait.pid) wait_n80=$(cat /root/logs/p3804_r757_wait_n80.pid)"
