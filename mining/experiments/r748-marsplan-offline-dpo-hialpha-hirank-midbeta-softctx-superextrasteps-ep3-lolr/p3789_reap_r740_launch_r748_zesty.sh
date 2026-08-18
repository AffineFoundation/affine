#!/usr/bin/env bash
# p3789: R740 REFUTE → reap chall 4,5 by pidfile → launch R748 TRAIN + wait→merge. Never pkill -f. Leave R747 on 6,7.
set -euo pipefail
log() { echo "[p3789-reap-r740] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R740 chall; keep /tmp/r740_merged; leave R747 TRAIN on 6,7"
stop_pidfile /root/logs/r740_sim_wvk7.pid "r740 sim"
stop_pidfile /root/logs/p3787_r740_lean.outer.pid "r740 lean outer"
stop_pidfile /root/logs/vllm_chall_r740.pid "r740 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r740 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r740_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 4,5 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r740_refute_reaped.p3789
log "launch R748 TRAIN"
nohup bash /root/mining_src/r748-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-superextrasteps-ep3-lolr/lean_train_zesty_gpus45_p3789.sh \
  >/root/logs/p3789_r748_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3789_r748_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r748-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-superextrasteps-ep3-lolr/wait_r748_train_then_merge_p3789.sh \
  >/root/logs/p3789_r748_wait.nohup 2>&1 &
echo $! >/root/logs/p3789_r748_wait.pid
log "R748 lean outer=$(cat /root/logs/p3789_r748_lean_outer.pid) wait=$(cat /root/logs/p3789_r748_wait.pid)"
