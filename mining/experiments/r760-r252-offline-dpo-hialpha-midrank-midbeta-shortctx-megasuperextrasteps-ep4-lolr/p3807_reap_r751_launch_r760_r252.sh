#!/usr/bin/env bash
# p3807: R751 REFUTE → reap chall by pid → launch R760 TRAIN 6,7 + wait→merge + wait→n80.
# Keep R759 TRAIN 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3807-reap-r751] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R751 chall; keep /tmp/r751_merged; leave R759 TRAIN on 4,5"
stop_pidfile /root/logs/r751_sim_wvk7.pid "r751 sim"
stop_pidfile /root/logs/p3805_r751_lean.outer.pid "r751 lean outer"
stop_pidfile /root/logs/vllm_chall_r751.pid "r751 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r751 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r751_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r751 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r252_gpus67_p3805\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r751_refute_reaped.p3807
EXP=r760-r252-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-lolr
log "launch R760 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3807.sh \
  >/root/logs/p3807_r760_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3807_r760_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r760_train_then_merge_p3807.sh \
  >/root/logs/p3807_r760_wait.nohup 2>&1 &
echo $! >/root/logs/p3807_r760_wait.pid
nohup bash /root/mining_src/$EXP/wait_r760_merge_then_n80_p3807.sh \
  >/root/logs/p3807_r760_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3807_r760_wait_n80.pid
log "R760 lean=$(cat /root/logs/p3807_r760_lean_outer.pid) wait=$(cat /root/logs/p3807_r760_wait.pid) wait_n80=$(cat /root/logs/p3807_r760_wait_n80.pid)"
