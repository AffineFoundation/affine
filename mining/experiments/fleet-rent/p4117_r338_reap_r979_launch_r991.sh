#!/usr/bin/env bash
# p4117: R979 REFUTE → exact-PID reap chall :8003 → R991 TRAIN on GPUs 4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R990 TRAIN 6,7.
set -euo pipefail
LOG=/root/logs/p4117_r338_reap_r979_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4117b] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R979 → R991"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4117b] kill pid=$pid ($why)"
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
  /root/logs/r979_sim_wvk7.pid \
  /root/logs/p4116_r979_lean_outer.pid \
  /root/logs/vllm_chall_r979.pid
do
  stop_pidfile "$pf" "stale outer/sim/chall"
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r979_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r979_merged/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4117b] wait free GPUs u45=$u45 iter=$i"
  [[ "$u45" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy u45=$u45"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4117b] TK still warm"

if [[ -f /root/logs/r990_train.pid ]] && kill -0 "$(cat /root/logs/r990_train.pid)" 2>/dev/null; then
  echo "[p4117b] R990 TRAIN still alive pid=$(cat /root/logs/r990_train.pid) (keep)"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4117_r979_refute_r991_armed.done
bash /root/mining_src/r991-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-hilr/lean_train_r338_gpus45_p4117.sh
echo "[p4117b] DONE armed R991"
