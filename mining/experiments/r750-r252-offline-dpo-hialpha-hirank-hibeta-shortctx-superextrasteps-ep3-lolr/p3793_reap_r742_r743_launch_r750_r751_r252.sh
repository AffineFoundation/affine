#!/usr/bin/env bash
# p3793: R742+R743 REFUTE → reap both challs by pid → launch R750 TRAIN 4,5 + R751 TRAIN 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3793-reap-r742-r743] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R742+R743 challs; keep /tmp/r742_merged /tmp/r743_merged"
stop_pidfile /root/logs/r742_sim_wvk7.pid "r742 sim"
stop_pidfile /root/logs/r743_sim_wvk7.pid "r743 sim"
stop_pidfile /root/logs/p3791_r742_lean.outer.pid "r742 lean outer"
stop_pidfile /root/logs/p3791_r743_lean.outer.pid "r743 lean outer"
stop_pidfile /root/logs/vllm_chall_r742.pid "r742 vllm"
stop_pidfile /root/logs/vllm_chall_r743.pid "r743 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r742 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r742_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r743 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r743_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r742/r743 lean bash"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r252_gpus(45|67)_p3791\.sh/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 used67=$used67 iter=$i"
  [[ "${used45:-999999}" -lt 8192 && "${used67:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
used67=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
[[ "${used67:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r742_refute_reaped.p3793
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r743_refute_reaped.p3793
log "launch R750 TRAIN 4,5 + R751 TRAIN 6,7"
nohup bash /root/mining_src/r750-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-superextrasteps-ep3-lolr/lean_train_r252_gpus45_p3793.sh \
  >/root/logs/p3793_r750_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3793_r750_lean_outer.pid
nohup bash /root/mining_src/r751-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-megasuperextrasteps-ep4-lolr/lean_train_r252_gpus67_p3793.sh \
  >/root/logs/p3793_r751_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3793_r751_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r750-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-superextrasteps-ep3-lolr/wait_r750_train_then_merge_p3793.sh \
  >/root/logs/p3793_r750_wait.nohup 2>&1 &
echo $! >/root/logs/p3793_r750_wait.pid
nohup bash /root/mining_src/r751-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-megasuperextrasteps-ep4-lolr/wait_r751_train_then_merge_p3793.sh \
  >/root/logs/p3793_r751_wait.nohup 2>&1 &
echo $! >/root/logs/p3793_r751_wait.pid
log "R750 lean=$(cat /root/logs/p3793_r750_lean_outer.pid) wait=$(cat /root/logs/p3793_r750_wait.pid)"
log "R751 lean=$(cat /root/logs/p3793_r751_lean_outer.pid) wait=$(cat /root/logs/p3793_r751_wait.pid)"
