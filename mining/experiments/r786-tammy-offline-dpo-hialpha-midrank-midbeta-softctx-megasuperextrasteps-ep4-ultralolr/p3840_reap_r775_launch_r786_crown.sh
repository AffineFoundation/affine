#!/usr/bin/env bash
# p3840: R775 REFUTE → reap chall :8002 on 4,5 → R786 Soft MidRank MidBeta SoftCtx Mega UltraLoLR TRAIN.
# Keep R776 MERGE/n80 on 6,7. Keep /tmp/r775_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3840-reap-r775] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R775 chall/n80; keep R776 on 6,7; keep /tmp/r775_merged"
stop_pidfile /root/logs/vllm_chall_r775.pid "r775 vllm"
stop_pidfile /root/logs/r775_sim_wvk7.pid "r775 sim"
stop_pidfile /root/logs/p3823_r775_lean.outer.pid "r775 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r775 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r775_merged|run_sim_duel.py.*r775|lean_chall_n80_crown_gpus45_p3823/ && !/awk/ {print $1}')
# also free :8002 listeners belonging to r775
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r775_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r776' && continue
  stop_pid "$pid" "r775 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r775_refute_reaped.p3840
EXP=r786-tammy-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R786 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus45_p3840.sh   >/root/logs/p3840_r786_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3840_r786_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r786_train_then_merge_p3840.sh   >/root/logs/p3840_r786_wait.nohup 2>&1 &
echo $! >/root/logs/p3840_r786_wait.pid
nohup bash /root/mining_src/$EXP/wait_r786_merge_then_n80_p3840.sh   >/root/logs/p3840_r786_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3840_r786_wait_n80.pid
log "R786 lean=$(cat /root/logs/p3840_r786_lean_outer.pid) wait=$(cat /root/logs/p3840_r786_wait.pid) wait_n80=$(cat /root/logs/p3840_r786_wait_n80.pid)"
