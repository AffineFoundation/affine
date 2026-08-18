#!/usr/bin/env bash
# p3855: R786 REFUTE → reap chall :8002 on 4,5 → R798 MidCtx MidRank MidBeta Mega UltraLoLR TRAIN.
# Keep R799 launch on 6,7. Keep /tmp/r786_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3855-reap-r786] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R786 chall/n80; keep R799 path on 6,7; keep /tmp/r786_merged"
stop_pidfile /root/logs/vllm_chall_r786.pid "r786 vllm"
stop_pidfile /root/logs/r786_sim_wvk7.pid "r786 sim"
stop_pidfile /root/logs/p3840_r786_lean.outer.pid "r786 lean outer"
stop_pidfile /root/logs/p3840_r786_lean_outer.pid "r786 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r786 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r786_merged|run_sim_duel.py.*r786|lean_chall_n80_crown_gpus45_p3840/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r786_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r799|r788' && continue
  stop_pid "$pid" "r786 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r786_refute_reaped.p3855
EXP=r798-tammy-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R798 TRAIN 4,5 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus45_p3855.sh   >/root/logs/p3855_r798_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3855_r798_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r798_train_then_merge_p3855.sh   >/root/logs/p3855_r798_wait.nohup 2>&1 &
echo $! >/root/logs/p3855_r798_wait.pid
nohup bash /root/mining_src/$EXP/wait_r798_merge_then_n80_p3855.sh   >/root/logs/p3855_r798_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3855_r798_wait_n80.pid
log "R798 lean=$(cat /root/logs/p3855_r798_lean_outer.pid) wait=$(cat /root/logs/p3855_r798_wait.pid) wait_n80=$(cat /root/logs/p3855_r798_wait_n80.pid)"
