#!/usr/bin/env bash
# p3792: R738 REFUTE → reap chall 6,7 by pidfile → launch R749 TRAIN + wait→merge. Never pkill -f. Leave R741 on 4,5.
set -euo pipefail
log() { echo "[p3792-reap-r738] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R738 chall; keep /tmp/r738_merged; leave R741 TRAIN on 4,5"
stop_pidfile /root/logs/r738_sim_wvk7.pid "r738 sim"
stop_pidfile /root/logs/p3790_r738_lean.outer.pid "r738 lean outer"
stop_pidfile /root/logs/vllm_chall_r738.pid "r738 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r738 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r738_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 6,7 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r738_refute_reaped.p3792
log "launch R749 TRAIN"
nohup bash /root/mining_src/r749-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_train_lunar_gpus67_p3792.sh \
  >/root/logs/p3792_r749_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3792_r749_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r749-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-lolr/wait_r749_train_then_merge_p3792.sh \
  >/root/logs/p3792_r749_wait.nohup 2>&1 &
echo $! >/root/logs/p3792_r749_wait.pid
log "R749 lean outer=$(cat /root/logs/p3792_r749_lean_outer.pid) wait=$(cat /root/logs/p3792_r749_wait.pid)"
