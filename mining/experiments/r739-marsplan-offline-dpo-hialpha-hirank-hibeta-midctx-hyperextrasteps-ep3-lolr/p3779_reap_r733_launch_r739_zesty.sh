#!/usr/bin/env bash
# p3779: R733 REFUTE → reap chall 6,7 by pidfile → launch R739 TRAIN + wait→merge. Never pkill -f.
set -euo pipefail
log() { echo "[p3779-reap-r733] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R733 chall; keep /tmp/r733_merged"
stop_pidfile /root/logs/r733_sim_wvk7.pid "r733 sim"
stop_pidfile /root/logs/p3778_r733_chall_outer.pid "r733 outer"
stop_pidfile /root/logs/vllm_chall_r733.pid "r733 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r733 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r733_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 6,7 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r733_refute_reaped.p3779
log "launch R739 TRAIN"
nohup bash /root/mining_src/r739-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-hyperextrasteps-ep3-lolr/lean_train_zesty_gpus67_p3779.sh \
  >/root/logs/p3779_r739_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3779_r739_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r739-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-hyperextrasteps-ep3-lolr/wait_r739_train_then_merge_p3779.sh \
  >/root/logs/p3779_r739_wait.nohup 2>&1 &
echo $! >/root/logs/p3779_r739_wait.pid
log "R739 lean outer=$(cat /root/logs/p3779_r739_lean_outer.pid) wait=$(cat /root/logs/p3779_r739_wait.pid)"
