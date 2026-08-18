#!/usr/bin/env bash
# p3869: R781 REFUTE ~−0.14× → reap chall :8002 on 4,5 → R805 Soft Hi Hi Soft UltraLoLR TRAIN.
# Leave R802 TRAIN/wait on 6,7 alone. Never pkill -f. Keep /tmp/r781_merged.
set -euo pipefail
log() { echo "[p3869-reap-r781] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R781 chall/n80 on 4,5; keep /tmp/r781_merged; leave R802 on 6,7"
stop_pidfile /root/logs/vllm_chall_r781.pid "r781 vllm"
stop_pidfile /root/logs/r781_sim_wvk7.pid "r781 sim"
stop_pidfile /root/logs/p3848_r781_chall_n80.pid "r781 lean outer"
stop_pidfile /root/logs/p3848_r781_lean_outer.pid "r781 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r781 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r781_merged|run_sim_duel.py.*r781|lean_chall_n80_r252_gpus45/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r781_merged\|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r802|r805' && continue
  stop_pid "$pid" "r781 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r781_refute_reaped.p3869
EXP=r805-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r805 /root/mining_src/$EXP
if [[ -s /root/mining_src/r667-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r667-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl /root/r805/dpo_duel_reason.jsonl
elif [[ -s /root/mining_src/r602-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r602-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps/dpo_duel_reason.jsonl /root/r805/dpo_duel_reason.jsonl
fi
cp -f /root/r805/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl 2>/dev/null || true
log "launch R805 TRAIN 4,5 + wait→merge + wait→n80 (:8002)"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3869.sh >/root/logs/p3869_r805_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3869_r805_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r805_train_then_merge_p3869.sh >/root/logs/p3869_r805_wait.nohup 2>&1 &
echo $! >/root/logs/p3869_r805_wait.pid
nohup bash /root/mining_src/$EXP/wait_r805_merge_then_n80_p3869.sh >/root/logs/p3869_r805_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3869_r805_wait_n80.pid
log "R805 lean=$(cat /root/logs/p3869_r805_lean_outer.pid) wait=$(cat /root/logs/p3869_r805_wait.pid) wait_n80=$(cat /root/logs/p3869_r805_wait_n80.pid)"
