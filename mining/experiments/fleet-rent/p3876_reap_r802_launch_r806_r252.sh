#!/usr/bin/env bash
# p3876: R802 REFUTE ~−0.47× → reap chall :8003 on 6,7 → R806 Soft Mid Hi Soft Mega UltraLoLR TRAIN.
# Leave R805 TRAIN on 4,5 alone. Never pkill -f. Keep /tmp/r802_merged.
set -euo pipefail
log() { echo "[p3876-reap-r802] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R802 chall/n80 on 6,7; keep /tmp/r802_merged; free :8003 for R806"
stop_pidfile /root/logs/vllm_chall_r802.pid "r802 vllm"
stop_pidfile /root/logs/r802_sim_wvk7.pid "r802 sim"
stop_pidfile /root/logs/p3860_r802_chall_n80.pid "r802 lean outer"
stop_pidfile /root/logs/p3860_r802_lean_outer.pid "r802 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r802 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r802_merged|run_sim_duel.py.*r802|lean_chall_n80_r252_gpus67_p3860/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r802_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r805|r806' && continue
  stop_pid "$pid" "r802 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r802_refute_reaped.p3876
EXP=r806-r252-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r806 /root/mining_src/$EXP
if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r806/dpo_duel_reason.jsonl
elif [[ -s /root/mining_src/r762-r252-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-lolr/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r762-r252-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-lolr/dpo_duel_reason.jsonl /root/r806/dpo_duel_reason.jsonl
elif [[ -s /root/mining_src/r711-r252-offline-dpo-hialpha-midrank-hibeta-softctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
  cp -f /root/mining_src/r711-r252-offline-dpo-hialpha-midrank-hibeta-softctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl /root/r806/dpo_duel_reason.jsonl
fi
log "launch R806 TRAIN 6,7 + wait→merge + wait→n80 (:8003)"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3876.sh >/root/logs/p3876_r806_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3876_r806_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r806_train_then_merge_p3876.sh >/root/logs/p3876_r806_wait.nohup 2>&1 &
echo $! >/root/logs/p3876_r806_wait.pid
nohup bash /root/mining_src/$EXP/wait_r806_merge_then_n80_p3876.sh >/root/logs/p3876_r806_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3876_r806_wait_n80.pid
log "R806 lean=$(cat /root/logs/p3876_r806_lean_outer.pid) wait=$(cat /root/logs/p3876_r806_wait.pid) wait_n80=$(cat /root/logs/p3876_r806_wait_n80.pid)"
