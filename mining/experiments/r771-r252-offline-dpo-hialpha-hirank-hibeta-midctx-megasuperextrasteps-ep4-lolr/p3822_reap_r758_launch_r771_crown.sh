#!/usr/bin/env bash
# p3822: R758 REFUTE → reap chall by pid → launch R771 TRAIN 4,5 + wait→merge + wait→n80.
# Keep R763 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3822-reap-r758] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R758 chall; keep /tmp/r758_merged; leave R763 TRAIN on 6,7"
stop_pidfile /root/logs/r758_sim_wvk7.pid "r758 sim"
stop_pidfile /root/logs/p3804_r758_lean.outer.pid "r758 lean outer"
stop_pidfile /root/logs/vllm_chall_r758.pid "r758 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r758 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r758_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r758 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_gpus45_p3804\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r758 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r758/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r758_refute_reaped.p3822
EXP=r771-r252-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-lolr
log "launch R771 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus45_p3822.sh   >/root/logs/p3822_r771_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3822_r771_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r771_train_then_merge_p3822.sh   >/root/logs/p3822_r771_wait.nohup 2>&1 &
echo $! >/root/logs/p3822_r771_wait.pid
nohup bash /root/mining_src/$EXP/wait_r771_merge_then_n80_p3822.sh   >/root/logs/p3822_r771_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3822_r771_wait_n80.pid
log "R771 lean=$(cat /root/logs/p3822_r771_lean_outer.pid) wait=$(cat /root/logs/p3822_r771_wait.pid) wait_n80=$(cat /root/logs/p3822_r771_wait_n80.pid)"
