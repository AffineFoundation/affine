#!/usr/bin/env bash
# p3886: R792 REFUTE ~−0.38× → reap chall :8002 on 4,5 → R814 MidCtx HiRank LoBeta UltraLoLR TRAIN.
# Leave R803 TRAIN/wait on 6,7 alone. Never pkill -f. Keep /tmp/r792_merged.
set -euo pipefail
log() { echo "[p3886-reap-r792] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R792 chall/n80 on 4,5; keep /tmp/r792_merged; leave R803 on 6,7"
stop_pidfile /root/logs/vllm_chall_r792.pid "r792 vllm"
stop_pidfile /root/logs/r792_sim_wvk7.pid "r792 sim"
stop_pidfile /root/logs/p3845_r792_chall_n80.pid "r792 lean outer"
stop_pidfile /root/logs/p3845_r792_lean_outer.pid "r792 lean outer"
stop_pidfile /root/logs/p3845_r792_lean.outer.pid "r792 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r792 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r792_merged|run_sim_duel.py.*r792|lean_chall_n80_golden_gpus45_p3845/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r792_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r803|r814' && continue
  stop_pid "$pid" "r792 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r792_refute_reaped.p3886
EXP=r814-r252-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r814 /root/mining_src/$EXP
if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r814/dpo_duel_reason.jsonl
elif [[ -s /root/r755/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r755/dpo_duel_reason.jsonl /root/r814/dpo_duel_reason.jsonl
  cp -f /root/r814/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
elif [[ -s /root/r745/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r745/dpo_duel_reason.jsonl /root/r814/dpo_duel_reason.jsonl
fi
log "launch R814 TRAIN 4,5 + wait→merge + wait→n80 (:8002)"
nohup bash /root/mining_src/$EXP/lean_train_golden_gpus45_p3886.sh >/root/logs/p3886_r814_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r814_train_then_merge_p3886.sh >/root/logs/p3886_r814_wait.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_wait.pid
nohup bash /root/mining_src/$EXP/wait_r814_merge_then_n80_p3886.sh >/root/logs/p3886_r814_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_wait_n80.pid
log "R814 lean=$(cat /root/logs/p3886_r814_lean_outer.pid) wait=$(cat /root/logs/p3886_r814_wait.pid) wait_n80=$(cat /root/logs/p3886_r814_wait_n80.pid)"
