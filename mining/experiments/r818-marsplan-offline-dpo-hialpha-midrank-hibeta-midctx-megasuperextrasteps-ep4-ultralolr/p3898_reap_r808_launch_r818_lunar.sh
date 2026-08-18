#!/usr/bin/env bash
# p3898: R808 REFUTE ~0.16× → reap chall :8003 GPUs 6,7 → R818 MidCtx MidRank HiBeta UltraLoLR TRAIN.
# Leave R817 TRAIN on 4,5. Never pkill -f. Keep /tmp/r808_merged.
set -euo pipefail
log() { echo "[p3898-reap-r808] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R808 chall :8003 on 6,7; leave R817 on 4,5; keep /tmp/r808_merged"
stop_pid 796664 "r808 vllm :8003"
stop_pid 798991 "r808 sim (if still)"
stop_pidfile /root/logs/r808_sim_wvk7.pid "r808 sim"
stop_pidfile /root/logs/vllm_chall_r808.pid "r808 vllm pidf"
stop_pidfile /root/logs/p3877_r808_chall_n80.pid "r808 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r808 vllm/sim/lean argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r808_merged|run_sim_duel.py.*r808|lean_chall_n80_lunar_gpus67_p3877/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r808_merged|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r817|train_dpo' && continue
  stop_pid "$pid" "r808 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r808_refute_reaped.p3898

EXP=r818-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r818 /root/mining_src/$EXP
if [[ ! -s /root/r818/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r818/dpo_duel_reason.jsonl
  elif [[ -s /root/r807/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r807/dpo_duel_reason.jsonl /root/r818/dpo_duel_reason.jsonl
  elif [[ -s /root/r817/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r817/dpo_duel_reason.jsonl /root/r818/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r818/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
log "launch R818 TRAIN 6,7 + wait→merge + wait→n80 (:8003); leave R817"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus67_p3898.sh >/root/logs/p3898_r818_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3898_r818_lean_outer.pid
sleep 1
nohup bash /root/mining_src/$EXP/wait_r818_train_then_merge_p3898.sh >/root/logs/p3898_r818_wait.nohup 2>&1 &
echo $! >/root/logs/p3898_r818_wait.pid
nohup bash /root/mining_src/$EXP/wait_r818_merge_then_n80_p3898.sh >/root/logs/p3898_r818_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3898_r818_wait_n80.pid
log "R818 lean=$(cat /root/logs/p3898_r818_lean_outer.pid) wait=$(cat /root/logs/p3898_r818_wait.pid) n80w=$(cat /root/logs/p3898_r818_wait_n80.pid)"
for i in $(seq 1 45); do
  if [[ -f /root/logs/r818_train.pid ]]; then
    tpid=$(cat /root/logs/r818_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_OK pid=$tpid"
      tr '\0' ' ' </proc/$tpid/cmdline; echo
      exit 0
    fi
  fi
  sleep 2
done
log "WARN train pid not yet visible; check lean outer log"
tail -40 /root/logs/p3898_r818_lean_outer.nohup || true
tail -40 /root/logs/r818_lean_warm.log || true
exit 1
