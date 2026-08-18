#!/usr/bin/env bash
# p3855: R788 REFUTE → reap chall :8003 on 6,7 → R799 MidCtx HiRank MidBeta Mega UltraLoLR TRAIN.
# Keep R798 TRAIN on 4,5. Keep /tmp/r788_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3855-reap-r788] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R788 chall/n80; keep R798 on 4,5; keep /tmp/r788_merged"
stop_pidfile /root/logs/vllm_chall_r788.pid "r788 vllm"
stop_pidfile /root/logs/r788_sim_wvk7.pid "r788 sim"
stop_pidfile /root/logs/p3841_r788_lean.outer.pid "r788 lean outer"
stop_pidfile /root/logs/p3841_r788_lean_outer.pid "r788 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r788 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r788_merged|run_sim_duel.py.*r788|lean_chall_n80_crown_gpus67_p3841/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r788_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r798|r786' && continue
  stop_pid "$pid" "r788 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r788_refute_reaped.p3855
EXP=r799-tammy-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
log "launch R799 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus67_p3855.sh   >/root/logs/p3855_r799_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3855_r799_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r799_train_then_merge_p3855.sh   >/root/logs/p3855_r799_wait.nohup 2>&1 &
echo $! >/root/logs/p3855_r799_wait.pid
nohup bash /root/mining_src/$EXP/wait_r799_merge_then_n80_p3855.sh   >/root/logs/p3855_r799_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3855_r799_wait_n80.pid
log "R799 lean=$(cat /root/logs/p3855_r799_lean_outer.pid) wait=$(cat /root/logs/p3855_r799_wait.pid) wait_n80=$(cat /root/logs/p3855_r799_wait_n80.pid)"
