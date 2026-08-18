#!/usr/bin/env bash
# p3886: R803 REFUTE ~0.60× + R804 REFUTE ~0.44× → reap challs :8003/:8002 → R813+R814 TRAIN.
# Never pkill -f. Keep /tmp/r803_merged /tmp/r804_merged.
set -euo pipefail
log() { echo "[p3886-reap-r803-r804] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R803(:8003)+R804(:8002); keep merges; launch R813(6,7)+R814(4,5)"
# exact known vllm parents
stop_pid 635460 "r803 vllm :8003"
stop_pid 641468 "r804 vllm :8002"
stop_pidfile /root/logs/r803_sim_wvk7.pid "r803 sim"
stop_pidfile /root/logs/r804_sim_wvk7.pid "r804 sim"
stop_pidfile /root/logs/vllm_chall_r803.pid "r803 vllm pidf"
stop_pidfile /root/logs/vllm_chall_r804.pid "r804 vllm pidf"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r803/r804 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r80[34]_merged|run_sim_duel.py.*r80[34]/ && !/awk/ {print $1}')
for port in 8002 8003; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    echo "$cmd" | grep -qE "r80[34]_merged|port ${port}" || continue
    echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r813|r814' && continue
    stop_pid "$pid" "port $port leftover"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
done
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 used67=$used67 iter=$i"
  [[ "${used45:-999999}" -lt 8192 && "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 && "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r803_refute_reaped.p3886
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r804_refute_reaped.p3886

EXP813=r813-r252-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr
EXP814=r814-r252-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r813 /root/r814 /root/mining_src/$EXP813 /root/mining_src/$EXP814
# MidCtx data from prior
for hyp in r813 r814; do
  if [[ ! -s /root/$hyp/dpo_duel_reason.jsonl ]]; then
    if [[ -s /root/r804/dpo_duel_reason.jsonl ]]; then
      cp -f /root/r804/dpo_duel_reason.jsonl /root/$hyp/dpo_duel_reason.jsonl
    elif [[ -s /root/r803/dpo_duel_reason.jsonl ]]; then
      cp -f /root/r803/dpo_duel_reason.jsonl /root/$hyp/dpo_duel_reason.jsonl
    fi
    cp -f /root/$hyp/dpo_duel_reason.jsonl /root/mining_src/r${hyp#r}*/dpo_duel_reason.jsonl 2>/dev/null || \
      cp -f /root/$hyp/dpo_duel_reason.jsonl /root/mining_src/$EXP813/dpo_duel_reason.jsonl
  fi
done
cp -f /root/r813/dpo_duel_reason.jsonl /root/mining_src/$EXP813/dpo_duel_reason.jsonl
cp -f /root/r814/dpo_duel_reason.jsonl /root/mining_src/$EXP814/dpo_duel_reason.jsonl

log "launch R813 TRAIN 6,7 + wait→merge + wait→n80 (:8003)"
nohup bash /root/mining_src/$EXP813/lean_train_golden_gpus67_p3886.sh >/root/logs/p3886_r813_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_lean_outer.pid
sleep 1
nohup bash /root/mining_src/$EXP813/wait_r813_train_then_merge_p3886.sh >/root/logs/p3886_r813_wait.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_wait.pid
nohup bash /root/mining_src/$EXP813/wait_r813_merge_then_n80_p3886.sh >/root/logs/p3886_r813_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_wait_n80.pid

log "launch R814 TRAIN 4,5 + wait→merge + wait→n80 (:8002)"
nohup bash /root/mining_src/$EXP814/lean_train_golden_gpus45_p3886.sh >/root/logs/p3886_r814_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_lean_outer.pid
sleep 1
nohup bash /root/mining_src/$EXP814/wait_r814_train_then_merge_p3886.sh >/root/logs/p3886_r814_wait.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_wait.pid
nohup bash /root/mining_src/$EXP814/wait_r814_merge_then_n80_p3886.sh >/root/logs/p3886_r814_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3886_r814_wait_n80.pid

log "R813 lean=$(cat /root/logs/p3886_r813_lean_outer.pid) wait=$(cat /root/logs/p3886_r813_wait.pid) n80w=$(cat /root/logs/p3886_r813_wait_n80.pid)"
log "R814 lean=$(cat /root/logs/p3886_r814_lean_outer.pid) wait=$(cat /root/logs/p3886_r814_wait.pid) n80w=$(cat /root/logs/p3886_r814_wait_n80.pid)"
