#!/usr/bin/env bash
# p3860: R793 REFUTE ~−0.55× → reap chall :8002 on 6,7 → R802 Soft Mid Lo Soft Mega UltraLoLR TRAIN.
# Leave R780 relay/waiter on 4,5 alone. Never pkill -f. Keep /tmp/r793_merged.
set -euo pipefail
log() { echo "[p3860-reap-r793] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R793 chall/n80 on 6,7; keep /tmp/r793_merged; free :8002 for R780"
stop_pidfile /root/logs/vllm_chall_r793.pid "r793 vllm"
stop_pidfile /root/logs/r793_sim_wvk7.pid "r793 sim"
stop_pidfile /root/logs/p3846_r793_chall_n80.pid "r793 lean outer"
stop_pidfile /root/logs/p3846_r793_lean_outer.pid "r793 lean outer"
stop_pidfile /root/logs/p3846_r793_lean.outer.pid "r793 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r793 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r793_merged|run_sim_duel.py.*r793|lean_chall_n80_r252_gpus67_p3846/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r793_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r780|r802' && continue
  stop_pid "$pid" "r793 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r793_refute_reaped.p3860
EXP=r802-r252-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
# Prefer Soft Mid Lo Soft data from r782 if r802 data is midbeta clone
if [[ -s /root/r782/dpo_duel_reason.jsonl ]]; then
  mkdir -p /root/r802 /root/mining_src/$EXP
  cp -f /root/r782/dpo_duel_reason.jsonl /root/r802/dpo_duel_reason.jsonl
  cp -f /root/r782/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
fi
log "launch R802 TRAIN 6,7 + wait→merge + wait→n80 (:8003)"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3860.sh >/root/logs/p3860_r802_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3860_r802_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r802_train_then_merge_p3860.sh >/root/logs/p3860_r802_wait.nohup 2>&1 &
echo $! >/root/logs/p3860_r802_wait.pid
nohup bash /root/mining_src/$EXP/wait_r802_merge_then_n80_p3860.sh >/root/logs/p3860_r802_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3860_r802_wait_n80.pid
log "R802 lean=$(cat /root/logs/p3860_r802_lean_outer.pid) wait=$(cat /root/logs/p3860_r802_wait.pid) wait_n80=$(cat /root/logs/p3860_r802_wait_n80.pid)"
