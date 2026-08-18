#!/usr/bin/env bash
# p3886: R791 REFUTE ~−1.32× → reap chall :8003 on 6,7 → R813 MidCtx Mid Hi UltraLoLR TRAIN.
# Leave R792 train/wait on 4,5 alone. Never pkill -f. Keep /tmp/r791_merged.
set -euo pipefail
log() { echo "[p3886-reap-r791] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R791 chall/n80 on 6,7; keep /tmp/r791_merged; leave R792 on 4,5"
stop_pidfile /root/logs/vllm_chall_r791.pid "r791 vllm"
stop_pidfile /root/logs/r791_sim_wvk7.pid "r791 sim"
stop_pidfile /root/logs/p3845_r791_chall_n80.pid "r791 lean outer"
stop_pidfile /root/logs/p3845_r791_lean_outer.pid "r791 lean outer"
stop_pidfile /root/logs/p3845_r791_lean.outer.pid "r791 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r791 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r791_merged|run_sim_duel.py.*r791|lean_chall_n80_golden_gpus67_p3845/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r791_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r792|r813' && continue
  stop_pid "$pid" "r791 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r791_refute_reaped.p3886
EXP=r813-r252-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr
# Prefer MidCtx HiBeta data from r729/r755 if present
mkdir -p /root/r813 /root/mining_src/$EXP
if [[ -s /root/mining_src/r729-r252-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r729-r252-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl /root/r813/dpo_duel_reason.jsonl
  cp -f /root/r813/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
elif [[ -s /root/r755/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r755/dpo_duel_reason.jsonl /root/r813/dpo_duel_reason.jsonl
  cp -f /root/r755/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
fi
log "launch R813 TRAIN 6,7 + wait→merge + wait→n80 (:8003)"
nohup bash /root/mining_src/$EXP/lean_train_golden_gpus67_p3886.sh >/root/logs/p3886_r813_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r813_train_then_merge_p3886.sh >/root/logs/p3886_r813_wait.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_wait.pid
nohup bash /root/mining_src/$EXP/wait_r813_merge_then_n80_p3886.sh >/root/logs/p3886_r813_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3886_r813_wait_n80.pid
log "R813 lean=$(cat /root/logs/p3886_r813_lean_outer.pid) wait=$(cat /root/logs/p3886_r813_wait.pid) wait_n80=$(cat /root/logs/p3886_r813_wait_n80.pid)"
