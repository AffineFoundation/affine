#!/usr/bin/env bash
# p3841: R777 REFUTE → reap chall :8003 on 6,7 → R787 Soft MidRank HiBeta SoftCtx Mega UltraLoLR TRAIN.
# Keep R785 TRAIN on 4,5. Keep /tmp/r777_merged. Never pkill -f.
set -euo pipefail
log() { echo "[p3841-reap-r777] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R777 chall/n80; keep R785 on 4,5; keep /tmp/r777_merged"
stop_pidfile /root/logs/vllm_chall_r777.pid "r777 vllm"
stop_pidfile /root/logs/r777_sim_wvk7.pid "r777 sim"
stop_pidfile /root/logs/p3827_r777_lean.outer.pid "r777 lean outer"
stop_pidfile /root/logs/p3827_r777_lean_outer.pid "r777 lean outer alt"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r777 vllm/sim argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r777_merged|run_sim_duel.py.*r777|lean_chall_n80_zesty_gpus67_p3827/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -q 'r777_merged\|port 8003' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r785' && continue
  stop_pid "$pid" "r777 :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r777_refute_reaped.p3841
EXP=r787-marsplan-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
log "launch R787 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3841.sh   >/root/logs/p3841_r787_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3841_r787_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r787_train_then_merge_p3841.sh   >/root/logs/p3841_r787_wait.nohup 2>&1 &
echo $! >/root/logs/p3841_r787_wait.pid
nohup bash /root/mining_src/$EXP/wait_r787_merge_then_n80_p3841.sh   >/root/logs/p3841_r787_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3841_r787_wait_n80.pid
log "R787 lean=$(cat /root/logs/p3841_r787_lean_outer.pid) wait=$(cat /root/logs/p3841_r787_wait.pid) wait_n80=$(cat /root/logs/p3841_r787_wait_n80.pid)"
