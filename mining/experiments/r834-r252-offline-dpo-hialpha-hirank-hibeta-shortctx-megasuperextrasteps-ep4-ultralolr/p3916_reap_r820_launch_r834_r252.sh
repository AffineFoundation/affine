#!/usr/bin/env bash
# p3916: R820 REFUTE ~−0.67× causality_fail → reap chall :8002 GPUs 4,5 → R834 Short Hi Hi UltraLoLR TRAIN.
# Leave R833 on 6,7/:8003. Never pkill -f. Keep /tmp/r820_merged.
set -euo pipefail
log() { echo "[p3916-reap-r820] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R820 chall :8002 on 4,5; leave R833 on 6,7; keep /tmp/r820_merged"
stop_pid 521716 "r820 sim"
stop_pid 519401 "r820 vllm :8002"
stop_pid 519272 "r820 lean_chall waiter"
stop_pidfile /root/logs/r820_sim_wvk7.pid "r820 sim"
stop_pidfile /root/logs/vllm_chall_r820.pid "r820 vllm pidf"
stop_pidfile /root/logs/p3914_r820_chall_n80.pid "r820 chall outer p3914"
stop_pidfile /root/logs/p3903_r820_chall_n80.pid "r820 chall outer p3903"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r820 vllm/sim/lean argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r820_merged|run_sim_duel.py.*r820|lean_chall_n80_r252_gpus45_p3903|lean_chall_n80_r252_gpus45_p3914/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r820_merged|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r833|r834' && continue
  stop_pid "$pid" "r820 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r820_refute_reaped.p3916

EXP=r834-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r834 /root/mining_src/$EXP
# Prefer Short Hi SoftCtx data from R820 / R825 / R730
if [[ ! -s /root/r834/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/r820/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r820/dpo_duel_reason.jsonl /root/r834/dpo_duel_reason.jsonl
  elif [[ -s /root/r825/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r825/dpo_duel_reason.jsonl /root/r834/dpo_duel_reason.jsonl
  elif [[ -s /root/r730/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r730/dpo_duel_reason.jsonl /root/r834/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r834/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl 2>/dev/null || true

# Ensure lean_chall exists with correct name (already in EXP dir from host sync)
chmod +x /root/mining_src/$EXP/*.sh
# Patch wait→n80 / wait→merge to correct chall/merge scripts if needed
sed -i 's|lean_chall_n80_r252_gpus45_p3905.sh|lean_chall_n80_r252_gpus45_p3916.sh|g' \
  /root/mining_src/$EXP/wait_r834_merge_then_n80_p3916.sh || true
sed -i 's|lean_merge_r252_gpus45_p3905.sh|lean_merge_r252_gpus45_p3916.sh|g' \
  /root/mining_src/$EXP/wait_r834_train_then_merge_p3916.sh || true

log "launch R834 lean train + waiters"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3916.sh >/root/logs/p3916_r834_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3916_r834_lean_outer.pid
nohup bash /root/mining_src/$EXP/wait_r834_train_then_merge_p3916.sh >/root/logs/p3916_r834_wait.nohup 2>&1 &
echo $! >/root/logs/p3916_r834_wait.pid
nohup bash /root/mining_src/$EXP/wait_r834_merge_then_n80_p3916.sh >/root/logs/p3916_r834_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3916_r834_wait_n80.pid
log "DONE lean_outer=$(cat /root/logs/p3916_r834_lean_outer.pid) wait=$(cat /root/logs/p3916_r834_wait.pid) n80wait=$(cat /root/logs/p3916_r834_wait_n80.pid)"
