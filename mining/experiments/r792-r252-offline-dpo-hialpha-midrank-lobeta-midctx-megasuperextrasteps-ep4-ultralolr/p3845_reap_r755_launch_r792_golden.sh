#!/usr/bin/env bash
# p3845: R755 REFUTE → reap chall → R792 TRAIN 4,5; keep R773 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3845-reap-r755] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"; kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1; local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid; pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"; rm -f "$pidf"
}
log "START reap R755 chall; keep /tmp/r755_merged; leave R773 TRAIN on 6,7"
stop_pidfile /root/logs/r755_sim_wvk7.pid "r755 sim"
stop_pidfile /root/logs/p3800_r755_lean.outer.pid "r755 lean outer"
stop_pidfile /root/logs/vllm_chall_r755.pid "r755 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r755 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r755_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r755 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_golden_gpus45_p3800\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r755 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r755/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r755_refute_reaped.p3845
EXP=r792-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
# locate lean_train script
TRAIN=$(ls /root/mining_src/$EXP/lean_train_golden_gpus45*.sh | head -1)
MERGE_WAIT=$(ls /root/mining_src/$EXP/wait_r792_train_then_merge*.sh | head -1)
N80_WAIT=$(ls /root/mining_src/$EXP/wait_r792_merge_then_n80*.sh | head -1)
log "launch R792 TRAIN 4,5 scripts=$TRAIN"
nohup bash "$TRAIN" >/root/logs/p3845_r792_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_lean_outer.pid
sleep 2
nohup bash "$MERGE_WAIT" >/root/logs/p3845_r792_wait.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_wait.pid
nohup bash "$N80_WAIT" >/root/logs/p3845_r792_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3845_r792_wait_n80.pid
log "R792 lean=$(cat /root/logs/p3845_r792_lean_outer.pid) wait=$(cat /root/logs/p3845_r792_wait.pid) wait_n80=$(cat /root/logs/p3845_r792_wait_n80.pid)"
