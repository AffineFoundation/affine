#!/usr/bin/env bash
# p3794: R741 REFUTE → reap chall by pid → launch R752 TRAIN 4,5. Keep R749 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3794-reap-r741] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R741 chall; keep /tmp/r741_merged; leave R749 on 6,7"
stop_pidfile /root/logs/r741_sim_wvk7.pid "r741 sim"
stop_pidfile /root/logs/p3792_r741_lean.outer.pid "r741 lean outer"
stop_pidfile /root/logs/vllm_chall_r741.pid "r741 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r741 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r741_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r741 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_lunar_gpus45_p3792\.sh/ && !/awk/ {print $1}')
# also kill any orphan children of vllm TP on GPUs 4,5 that belong to r741
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r741_refute_reaped.p3794
log "launch R752 TRAIN 4,5"
nohup bash /root/mining_src/r752-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_train_lunar_gpus45_p3794.sh \
  >/root/logs/p3794_r752_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3794_r752_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r752-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-lolr/wait_r752_train_then_merge_p3794.sh \
  >/root/logs/p3794_r752_wait.nohup 2>&1 &
echo $! >/root/logs/p3794_r752_wait.pid
log "R752 lean=$(cat /root/logs/p3794_r752_lean_outer.pid) wait=$(cat /root/logs/p3794_r752_wait.pid)"
