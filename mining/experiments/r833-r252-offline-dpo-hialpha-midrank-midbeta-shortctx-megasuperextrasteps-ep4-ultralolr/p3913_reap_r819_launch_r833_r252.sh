#!/usr/bin/env bash
# p3913: R819 REFUTE ~−1.57× → reap chall :8003 GPUs 6,7 → R833 Short Mid Mid Soft UltraLoLR TRAIN.
# Leave R820 on 4,5/:8002. Never pkill -f. Keep /tmp/r819_merged.
set -euo pipefail
log() { echo "[p3913-reap-r819] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R819 chall :8003 on 6,7; leave R820 on 4,5; keep /tmp/r819_merged"
stop_pid 516752 "r819 sim"
stop_pid 514535 "r819 vllm :8003"
stop_pid 514409 "r819 lean_chall waiter"
stop_pidfile /root/logs/r819_sim_wvk7.pid "r819 sim"
stop_pidfile /root/logs/vllm_chall_r819.pid "r819 vllm pidf"
stop_pidfile /root/logs/p3902_r819_chall_n80.pid "r819 chall outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r819 vllm/sim/lean argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r819_merged|run_sim_duel.py.*r819|lean_chall_n80_r252_gpus67_p3902/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r819_merged|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r820|r833' && continue
  stop_pid "$pid" "r819 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r819_refute_reaped.p3913

EXP=r833-r252-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r833 /root/mining_src/$EXP
if [[ ! -s /root/r833/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/r819/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r819/dpo_duel_reason.jsonl /root/r833/dpo_duel_reason.jsonl
  elif [[ -s /root/r770/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r770/dpo_duel_reason.jsonl /root/r833/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r833/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
# lean_chall from R819 chall with sed
src=/root/mining_src/r819-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p3902.sh
if [[ -f "$src" ]]; then
  sed -e 's/r819/r833/g' -e 's/R819/R833/g' -e 's/p3902/p3913/g' \
      -e 's/lobeta/midbeta/g' -e 's/LoBeta/MidBeta/g' \
      "$src" > /root/mining_src/$EXP/lean_chall_n80_r252_gpus67_p3913.sh
fi
chmod +x /root/mining_src/$EXP/*.sh
sed -i 's|lean_chall_n80_r252_gpus67_p3902.sh|lean_chall_n80_r252_gpus67_p3913.sh|g' \
  /root/mining_src/$EXP/wait_r833_merge_then_n80_p3913.sh || true

log "launch R833 lean train + waiters"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3913.sh >/root/logs/p3913_r833_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3913_r833_lean_outer.pid
nohup bash /root/mining_src/$EXP/wait_r833_train_then_merge_p3913.sh >/root/logs/p3913_r833_wait.nohup 2>&1 &
echo $! >/root/logs/p3913_r833_wait.pid
nohup bash /root/mining_src/$EXP/wait_r833_merge_then_n80_p3913.sh >/root/logs/p3913_r833_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3913_r833_wait_n80.pid
log "DONE lean_outer=$(cat /root/logs/p3913_r833_lean_outer.pid) wait=$(cat /root/logs/p3913_r833_wait.pid) n80wait=$(cat /root/logs/p3913_r833_wait_n80.pid)"
