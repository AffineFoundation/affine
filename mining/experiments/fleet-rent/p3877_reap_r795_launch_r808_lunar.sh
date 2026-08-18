#!/usr/bin/env bash
# p3877: R795 REFUTE ~0.045× → reap chall :8003 on lunar 6,7 → R808 Soft Hi Hi Soft Mega UltraLoLR TRAIN.
# Leave R807 TRAIN on 4,5 alone. Never pkill -f. Keep /tmp/r795_merged.
set -euo pipefail
log() { echo "[p3877-reap-r795] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R795 chall/n80 on 6,7; keep /tmp/r795_merged; free :8003 for R808"
stop_pidfile /root/logs/vllm_chall_r795.pid "r795 vllm"
stop_pidfile /root/logs/r795_sim_wvk7.pid "r795 sim"
stop_pidfile /root/logs/p3872_r795_chall_n80.pid "r795 lean outer"
stop_pidfile /root/logs/wait_r795_scp_then_chall_p3872.outer.pid "r795 wait outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r795 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r795_merged|run_sim_duel.py.*r795|lean_chall_n80_lunar_gpus67_p3872|wait_r795_scp_then_chall_p3872/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r795_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r807|r808|train_dpo' && continue
  stop_pid "$pid" "r795 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r795_refute_reaped.p3877
EXP=r808-marsplan-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r808 /root/mining_src/$EXP
if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r808/dpo_duel_reason.jsonl
elif [[ -s /root/mining_src/r794-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r794-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl /root/r808/dpo_duel_reason.jsonl
elif [[ -s /root/r807/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r807/dpo_duel_reason.jsonl /root/r808/dpo_duel_reason.jsonl
fi
log "launch R808 TRAIN 6,7 + wait→merge + wait→n80 (:8003)"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus67_p3877.sh >/root/logs/p3877_r808_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3877_r808_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r808_train_then_merge_p3877.sh >/root/logs/p3877_r808_wait.nohup 2>&1 &
echo $! >/root/logs/p3877_r808_wait.pid
nohup bash /root/mining_src/$EXP/wait_r808_merge_then_n80_p3877.sh >/root/logs/p3877_r808_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3877_r808_wait_n80.pid
log "R808 lean=$(cat /root/logs/p3877_r808_lean_outer.pid) wait=$(cat /root/logs/p3877_r808_wait.pid) wait_n80=$(cat /root/logs/p3877_r808_wait_n80.pid)"
