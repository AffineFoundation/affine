#!/usr/bin/env bash
# p3846: R782 REFUTE ~0.21x → reap chall :8002 on 6,7 → R793 Soft Mid Mid Soft Mega UltraLoLR TRAIN.
# Leave R762 waiter on 4,5 alone. Never pkill -f. Keep /tmp/r782_merged.
set -euo pipefail
log() { echo "[p3846-reap-r782] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R782 chall/n80 on 6,7; keep /tmp/r782_merged"
stop_pidfile /root/logs/vllm_chall_r782.pid "r782 vllm"
stop_pidfile /root/logs/r782_sim_wvk7.pid "r782 sim"
stop_pidfile /root/logs/p3833_r782_chall_n80.pid "r782 lean outer"
stop_pidfile /root/logs/p3833_r782_lean.outer.pid "r782 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r782 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r782_merged|run_sim_duel.py.*r782|lean_chall_n80_r252_gpus67_p3833/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r782_refute_reaped.p3846
EXP=r793-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R793 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3846.sh >/root/logs/p3846_r793_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3846_r793_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r793_train_then_merge_p3846.sh >/root/logs/p3846_r793_wait.nohup 2>&1 &
echo $! >/root/logs/p3846_r793_wait.pid
nohup bash /root/mining_src/$EXP/wait_r793_merge_then_n80_p3846.sh >/root/logs/p3846_r793_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3846_r793_wait_n80.pid
log "R793 lean=$(cat /root/logs/p3846_r793_lean_outer.pid) wait=$(cat /root/logs/p3846_r793_wait.pid) wait_n80=$(cat /root/logs/p3846_r793_wait_n80.pid)"
