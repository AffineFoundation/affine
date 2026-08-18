#!/usr/bin/env bash
# p3809: R753 REFUTE → reap chall by pid → launch R763 TRAIN 6,7 + wait→merge + wait→n80.
# Keep R758 TRAIN 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3809-reap-r753] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R753 chall; keep /tmp/r753_merged; leave R758 TRAIN on 4,5"
stop_pidfile /root/logs/r753_sim_wvk7.pid "r753 sim"
stop_pidfile /root/logs/p3806_r753_lean.outer.pid "r753 lean outer"
stop_pidfile /root/logs/vllm_chall_r753.pid "r753 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r753 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r753_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r753 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_gpus67_p3806\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r753_refute_reaped.p3809
EXP=r763-r252-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-lolr
log "launch R763 TRAIN 6,7 + wait→merge + wait→n80"
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus67_p3809.sh \
  >/root/logs/p3809_r763_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3809_r763_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r763_train_then_merge_p3809.sh \
  >/root/logs/p3809_r763_wait.nohup 2>&1 &
echo $! >/root/logs/p3809_r763_wait.pid
nohup bash /root/mining_src/$EXP/wait_r763_merge_then_n80_p3809.sh \
  >/root/logs/p3809_r763_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3809_r763_wait_n80.pid
log "R763 lean=$(cat /root/logs/p3809_r763_lean_outer.pid) wait=$(cat /root/logs/p3809_r763_wait.pid) wait_n80=$(cat /root/logs/p3809_r763_wait_n80.pid)"
