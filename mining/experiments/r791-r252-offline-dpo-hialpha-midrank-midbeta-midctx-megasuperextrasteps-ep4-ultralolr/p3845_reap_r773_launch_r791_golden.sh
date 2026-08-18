#!/usr/bin/env bash
# p3845: R773 REFUTE → reap chall :8003 on 6,7 → R791 MidCtx MidRank MidBeta Mega UltraLoLR TRAIN.
# Sibling R792 occupies 4,5. Never pkill -f. Keep /tmp/r773_merged.
set -euo pipefail
log() { echo "[p3845-reap-r773] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R773 chall/n80 on 6,7; keep /tmp/r773_merged"
stop_pidfile /root/logs/vllm_chall_r773.pid "r773 vllm"
stop_pidfile /root/logs/r773_sim_wvk7.pid "r773 sim"
stop_pidfile /root/logs/p3822_r773_lean.outer.pid "r773 lean outer"
stop_pidfile /root/logs/p3844_r773_lean.outer.pid "r773 lean outer p3844"
stop_pidfile /root/logs/p3822_r773_wait_n80.pid "r773 wait_n80"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r773 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r773_merged|run_sim_duel.py.*r773|lean_chall_n80_golden_gpus67/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r773_refute_reaped.p3845
EXP=r791-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R791 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_golden_gpus67_p3845.sh   >/root/logs/p3845_r791_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3845_r791_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r791_train_then_merge_p3845.sh   >/root/logs/p3845_r791_wait.nohup 2>&1 &
echo $! >/root/logs/p3845_r791_wait.pid
nohup bash /root/mining_src/$EXP/wait_r791_merge_then_n80_p3845.sh   >/root/logs/p3845_r791_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3845_r791_wait_n80.pid
log "R791 lean=$(cat /root/logs/p3845_r791_lean_outer.pid) wait=$(cat /root/logs/p3845_r791_wait.pid) wait_n80=$(cat /root/logs/p3845_r791_wait_n80.pid)"
