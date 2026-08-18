#!/usr/bin/env bash
# p3819: R760 REFUTE → reap chall by pid → launch R770 TRAIN 6,7 + wait→merge + wait→n80.
# Keep R769 TRAIN 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3819-reap-r760] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R760 chall; keep /tmp/r760_merged; leave R769 TRAIN on 4,5"
stop_pidfile /root/logs/r760_sim_wvk7.pid "r760 sim"
stop_pidfile /root/logs/p3807_r760_lean.outer.pid "r760 lean outer"
stop_pidfile /root/logs/vllm_chall_r760.pid "r760 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r760 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r760_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r760 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r252_gpus67_p3807\.sh/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r760 sim duel"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r760/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r760_refute_reaped.p3819
EXP=r770-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-lolr
log "launch R770 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3819.sh \
  >/root/logs/p3819_r770_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3819_r770_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r770_train_then_merge_p3819.sh \
  >/root/logs/p3819_r770_wait.nohup 2>&1 &
echo $! >/root/logs/p3819_r770_wait.pid
nohup bash /root/mining_src/$EXP/wait_r770_merge_then_n80_p3819.sh \
  >/root/logs/p3819_r770_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3819_r770_wait_n80.pid
log "R770 lean=$(cat /root/logs/p3819_r770_lean_outer.pid) wait=$(cat /root/logs/p3819_r770_wait.pid) wait_n80=$(cat /root/logs/p3819_r770_wait_n80.pid)"
