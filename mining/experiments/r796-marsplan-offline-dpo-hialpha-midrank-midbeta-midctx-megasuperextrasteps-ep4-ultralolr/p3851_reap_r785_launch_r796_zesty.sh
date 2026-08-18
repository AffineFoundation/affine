#!/usr/bin/env bash
# p3851: R785 REFUTE → reap chall :8002 on 4,5 → R796 MidCtx MidRank MidBeta Mega UltraLoLR TRAIN.
# Keep R787 TRAIN on 6,7. Keep /tmp/r785_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3851-reap-r785] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R785 chall/n80; keep R787 on 6,7; keep /tmp/r785_merged"
stop_pidfile /root/logs/vllm_chall_r785.pid "r785 vllm"
stop_pidfile /root/logs/r785_sim_wvk7.pid "r785 sim"
stop_pidfile /root/logs/p3839_r785_lean.outer.pid "r785 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r785 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r785_merged|run_sim_duel.py.*r785|lean_chall_n80_zesty_gpus45_p3839/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r785_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r787' && continue
  stop_pid "$pid" "r785 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r785_refute_reaped.p3851
EXP=r796-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R796 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus45_p3851.sh   >/root/logs/p3851_r796_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3851_r796_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r796_train_then_merge_p3851.sh   >/root/logs/p3851_r796_wait.nohup 2>&1 &
echo $! >/root/logs/p3851_r796_wait.pid
nohup bash /root/mining_src/$EXP/wait_r796_merge_then_n80_p3851.sh   >/root/logs/p3851_r796_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3851_r796_wait_n80.pid
log "R796 lean=$(cat /root/logs/p3851_r796_lean_outer.pid) wait=$(cat /root/logs/p3851_r796_wait.pid) wait_n80=$(cat /root/logs/p3851_r796_wait_n80.pid)"
