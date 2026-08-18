#!/usr/bin/env bash
# p3788: R739 REFUTE → reap chall 6,7 by pidfile → launch R747 TRAIN + wait→merge. Never pkill -f. Leave R740 on 4,5.
set -euo pipefail
log() { echo "[p3788-reap-r739] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R739 chall; keep /tmp/r739_merged; leave R740 n80 on 4,5"
stop_pidfile /root/logs/r739_sim_wvk7.pid "r739 sim"
stop_pidfile /root/logs/p3786_r739_lean.outer.pid "r739 lean outer"
stop_pidfile /root/logs/vllm_chall_r739.pid "r739 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r739 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r739_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 6,7 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r739_refute_reaped.p3788
log "launch R747 TRAIN"
nohup bash /root/mining_src/r747-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr/lean_train_zesty_gpus67_p3788.sh \
  >/root/logs/p3788_r747_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3788_r747_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r747-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr/wait_r747_train_then_merge_p3788.sh \
  >/root/logs/p3788_r747_wait.nohup 2>&1 &
echo $! >/root/logs/p3788_r747_wait.pid
log "R747 lean outer=$(cat /root/logs/p3788_r747_lean_outer.pid) wait=$(cat /root/logs/p3788_r747_wait.pid)"
