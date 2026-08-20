#!/usr/bin/env bash
# p4129: R988 REFUTE → exact-PID reap chall :8002 → R1002 TRAIN on GPUs 4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R998 TRAIN 6,7.
set -euo pipefail
LOG=/root/logs/p4129_r252_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4129-r252] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R988 → R1002"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4129-r252] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why pidf=$pidf"
  rm -f "$pidf"
}

for pf in \
  /root/logs/r988_sim_wvk7.pid \
  /root/logs/p4127_r988_lean_outer.pid \
  /root/logs/p4127_r988_outer.pid \
  /root/logs/vllm_chall_r988.pid
do
  stop_pidfile "$pf" "stale outer/sim/chall"
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r988_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r988_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r988"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r252_r988/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4129-r252] wait free GPUs u45=$u45 iter=$i"
  [[ "$u45" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy u45=$u45"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4129-r252] TK still warm"

if [[ -f /root/logs/r998_train.pid ]]; then
  r998=$(cat /root/logs/r998_train.pid)
  if kill -0 "$r998" 2>/dev/null; then
    echo "[p4129-r252] R998 TRAIN keep pid=$r998"
  fi
fi

chmod +x /root/mining_src/r1002-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/*.sh
bash /root/mining_src/r1002-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_train_r252_gpus45_p4129.sh
echo "[p4129-r252] R1002 pid=$(cat /root/logs/r1002_train.pid 2>/dev/null)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4129_r988_refute_r1002_armed.done
echo "[p4129-r252] DONE"
