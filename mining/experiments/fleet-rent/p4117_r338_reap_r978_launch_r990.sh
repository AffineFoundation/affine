#!/usr/bin/env bash
# p4117: R978 REFUTE → exact-PID reap chall :8002 → R990 TRAIN on GPUs 6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R979 n80 :8003 GPUs 4,5.
set -euo pipefail
LOG=/root/logs/p4117_r338_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4117] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R978 → R990"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4117] kill pid=$pid ($why)"
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
  /root/logs/r978_sim_wvk7.pid \
  /root/logs/p4115_r978_lean_outer.pid \
  /root/logs/p4115_r978_outer.pid \
  /root/logs/vllm_chall_r978.pid
do
  stop_pidfile "$pf" "stale outer/sim/chall"
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r978_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r978_merged/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4117] wait free GPUs u67=$u67 iter=$i"
  [[ "$u67" -lt 8192 ]] && break
  sleep 2
done
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u67" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy u67=$u67"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4117] TK still warm"

# Confirm R979 n80 still on 4,5 :8003
if ss -lptn 'sport = :8003' 2>/dev/null | grep -q LISTEN; then
  echo "[p4117] R979 chall :8003 still up (keep)"
else
  echo "[p4117] WARN :8003 not listening"
fi
if [[ -f /root/logs/r979_sim_wvk7.pid ]] || pgrep -f 'run_sim_duel.py.*r979' >/dev/null 2>&1; then
  echo "[p4117] R979 n80 still alive (keep)"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4117_r978_refute_r990_armed.done
bash /root/mining_src/r990-vera-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-hilr/lean_train_r338_gpus67_p4117.sh
echo "[p4117] DONE armed R990"
