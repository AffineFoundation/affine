#!/usr/bin/env bash
# p3903: R815 REFUTE ~−0.30× → reap chall :8002 GPUs 4,5 → R820 Short Hi Lo UltraLoLR TRAIN.
# Leave R819 TRAIN on 6,7. Never pkill -f. Keep /tmp/r815_merged.
set -euo pipefail
log() { echo "[p3903-reap-r815] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R815 chall :8002 on 4,5; leave R819 on 6,7; keep /tmp/r815_merged"
# exact known chall pid from inventory
stop_pid 505284 "r815 vllm :8002"
stop_pidfile /root/logs/r815_sim_wvk7.pid "r815 sim"
stop_pidfile /root/logs/vllm_chall_r815.pid "r815 vllm pidf"
stop_pidfile /root/logs/p3886_r815_chall_n80.pid "r815 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r815 vllm/sim/lean argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r815_merged|run_sim_duel.py.*r815|lean_chall_n80_r252_gpus45_p3886/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r815_merged|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r819|r820' && continue
  stop_pid "$pid" "r815 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r815_refute_reaped.p3903

EXP=r820-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r820 /root/mining_src/$EXP
# prefer Short Hi Lo Soft data (r743); fall back carefully
if [[ ! -s /root/r820/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r743-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-hyperextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r743-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-hyperextrasteps-ep3-lolr/dpo_duel_reason.jsonl /root/r820/dpo_duel_reason.jsonl
  elif [[ -s /root/r743/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r743/dpo_duel_reason.jsonl /root/r820/dpo_duel_reason.jsonl
  elif [[ -s /root/r815/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r815/dpo_duel_reason.jsonl /root/r820/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r820/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
chmod +x /root/mining_src/$EXP/*.sh
# ensure wait_n80 points at lean chall p3903
sed -i 's|lean_chall_n80_r252_gpus45_p3886.sh|lean_chall_n80_r252_gpus45_p3903.sh|g' \
  /root/mining_src/$EXP/wait_r820_merge_then_n80_p3903.sh || true
sed -i 's|lean_merge_r252_gpus45_p3886.sh|lean_merge_r252_gpus45_p3903.sh|g' \
  /root/mining_src/$EXP/wait_r820_train_then_merge_p3903.sh || true

log "launch R820 lean train + waiters"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3903.sh >/root/logs/p3903_r820_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3903_r820_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r820_train_then_merge_p3903.sh >/root/logs/p3903_r820_wait_merge.nohup 2>&1 &
echo $! >/root/logs/p3903_r820_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r820_merge_then_n80_p3903.sh >/root/logs/p3903_r820_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3903_r820_wait_n80.pid
log "DONE lean_outer=$(cat /root/logs/p3903_r820_lean_outer.pid) wait_merge=$(cat /root/logs/p3903_r820_wait_merge.pid) wait_n80=$(cat /root/logs/p3903_r820_wait_n80.pid)"
