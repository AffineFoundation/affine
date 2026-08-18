#!/usr/bin/env bash
# p3799: R747 REFUTE -> reap chall 6,7 -> R754 TRAIN + wait->merge + wait->n80. Never pkill -f. Leave R748 on 4,5.
set -euo pipefail
log() { echo "[p3799-reap-r747] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R747 chall; keep /tmp/r747_merged; leave R748 TRAIN on 4,5"
stop_pidfile /root/logs/r747_sim_wvk7.pid "r747 sim"
stop_pidfile /root/logs/p3797_r747_lean.outer.pid "r747 lean outer"
stop_pidfile /root/logs/vllm_chall_r747.pid "r747 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r747 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r747_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 6,7 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r747_refute_reaped.p3799
EXP=r754-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-lolr
log "launch R754 TRAIN + wait->merge + wait->n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3799.sh >/root/logs/p3799_r754_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3799_r754_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r754_train_then_merge_p3799.sh >/root/logs/p3799_r754_wait.nohup 2>&1 &
echo $! >/root/logs/p3799_r754_wait.pid
nohup bash /root/mining_src/$EXP/wait_r754_merge_then_n80_p3799.sh >/root/logs/p3799_r754_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3799_r754_wait_n80.pid
log "R754 lean=$(cat /root/logs/p3799_r754_lean_outer.pid) wait=$(cat /root/logs/p3799_r754_wait.pid) wait_n80=$(cat /root/logs/p3799_r754_wait_n80.pid)"
