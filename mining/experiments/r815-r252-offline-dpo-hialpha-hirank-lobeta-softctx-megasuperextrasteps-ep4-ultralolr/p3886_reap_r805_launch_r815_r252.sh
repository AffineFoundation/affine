#!/usr/bin/env bash
# p3886: R805 REFUTE ~−0.78× → reap chall :8002 GPUs 4,5 → R815 Soft Hi Lo Soft UltraLoLR TRAIN.
# Leave R806 TRAIN on 6,7. Never pkill -f. Keep /tmp/r805_merged.
set -euo pipefail
log() { echo "[p3886-reap-r805] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R805 chall :8002 on 4,5; leave R806 on 6,7; keep /tmp/r805_merged"
stop_pid 487599 "r805 vllm :8002"
stop_pidfile /root/logs/r805_sim_wvk7.pid "r805 sim"
stop_pidfile /root/logs/vllm_chall_r805.pid "r805 vllm pidf"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r805 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r805_merged|run_sim_duel.py.*r805/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r805_merged|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r806|r815' && continue
  stop_pid "$pid" "r805 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r805_refute_reaped.p3886

EXP=r815-r252-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r815 /root/mining_src/$EXP
if [[ ! -s /root/r815/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r553-r252-offline-dpo-hialpha-hirank-lobeta-softctx-extrasteps/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r553-r252-offline-dpo-hialpha-hirank-lobeta-softctx-extrasteps/dpo_duel_reason.jsonl /root/r815/dpo_duel_reason.jsonl
  elif [[ -s /root/r805/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r805/dpo_duel_reason.jsonl /root/r815/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r815/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
log "launch R815 TRAIN 4,5 + wait→merge + wait→n80 (:8002); leave R806"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3886.sh >/root/logs/p3886_r815_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3886_r815_lean_outer.pid
sleep 1
nohup bash /root/mining_src/$EXP/wait_r815_train_then_merge_p3886.sh >/root/logs/p3886_r815_wait.nohup 2>&1 &
echo $! >/root/logs/p3886_r815_wait.pid
nohup bash /root/mining_src/$EXP/wait_r815_merge_then_n80_p3886.sh >/root/logs/p3886_r815_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3886_r815_wait_n80.pid
log "R815 lean=$(cat /root/logs/p3886_r815_lean_outer.pid) wait=$(cat /root/logs/p3886_r815_wait.pid) n80w=$(cat /root/logs/p3886_r815_wait_n80.pid)"
