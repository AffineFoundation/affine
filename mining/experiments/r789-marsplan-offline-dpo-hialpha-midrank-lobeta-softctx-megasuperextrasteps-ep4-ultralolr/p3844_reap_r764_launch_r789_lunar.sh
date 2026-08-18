#!/usr/bin/env bash
# p3844: R764 REFUTE → reap chall :8003 on 6,7 → R789 Soft MidRank LoBeta SoftCtx Mega TRAIN.
# Keep R765 N80 on 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3844-reap-r764] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R764 chall/n80; keep R765 on 4,5; keep /tmp/r764_merged"
stop_pidfile /root/logs/vllm_chall_r764.pid "r764 vllm"
stop_pidfile /root/logs/r764_sim_wvk7.pid "r764 sim"
stop_pidfile /root/logs/p3809_r764_lean.outer.pid "r764 lean outer"
stop_pidfile /root/logs/p3809_r764_wait_n80.pid "r764 wait_n80"
stop_pidfile /root/logs/p3809_r764_wait.pid "r764 wait"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r764 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r764_merged|run_sim_duel.py.*r764|lean_chall_n80_lunar_gpus67_p3809/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r764_refute_reaped.p3844
EXP=r789-marsplan-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R789 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus67_p3844.sh   >/root/logs/p3844_r789_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r789_train_then_merge_p3844.sh   >/root/logs/p3844_r789_wait.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_wait.pid
nohup bash /root/mining_src/$EXP/wait_r789_merge_then_n80_p3844.sh   >/root/logs/p3844_r789_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_wait_n80.pid
log "R789 lean=$(cat /root/logs/p3844_r789_lean_outer.pid) wait=$(cat /root/logs/p3844_r789_wait.pid) wait_n80=$(cat /root/logs/p3844_r789_wait_n80.pid)"
