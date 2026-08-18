#!/usr/bin/env bash
# p3796: R744 REFUTE → reap chall by pid → launch R753 TRAIN 6,7. Keep R716 RELAY 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3796-reap-r744] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R744 chall; keep /tmp/r744_merged; leave R716 on 4,5"
stop_pidfile /root/logs/r744_sim_wvk7.pid "r744 sim"
stop_pidfile /root/logs/p3794_r744_lean.outer.pid "r744 lean outer"
stop_pidfile /root/logs/vllm_chall_r744.pid "r744 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r744 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r744_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r744 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_gpus67_p3794\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used67=$used67 iter=$i"
  [[ "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r744_refute_reaped.p3796
log "launch R753 TRAIN 6,7"
nohup bash /root/mining_src/r753-r252-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_train_crown_gpus67_p3796.sh \
  >/root/logs/p3796_r753_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3796_r753_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r753-r252-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-lolr/wait_r753_train_then_merge_p3796.sh \
  >/root/logs/p3796_r753_wait.nohup 2>&1 &
echo $! >/root/logs/p3796_r753_wait.pid
log "R753 lean=$(cat /root/logs/p3796_r753_lean_outer.pid) wait=$(cat /root/logs/p3796_r753_wait.pid)"
