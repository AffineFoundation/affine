#!/usr/bin/env bash
# p4143: R997 REFUTE → exact-PID reap chall :8003 → R1011 TRAIN on GPUs 4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1004 TRAIN 6,7.
set -euo pipefail
LOG=/root/logs/p4143_r337_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4143] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R997 → R1011"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4143] kill pid=$pid ($why)"
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
  /root/logs/r997_sim_wvk7.pid \
  /root/logs/p4134_r997_lean_outer.pid \
  /root/logs/p4134_r997_outer.pid \
  /root/logs/p4134_r997_merge_then_n80.outer.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

# stop lean_chall outer if still running
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r997"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r337_gpus45_p4134|lean_chall_n80.*r997/ && !/awk/ {print $1}')

stop_pidfile /root/logs/vllm_chall_r997.pid "r997 chall"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r997_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r997_merged/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4143] wait free GPUs u45=$u45 iter=$i"
  [[ "$u45" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy u45=$u45"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4143] TK still warm"

# Confirm R1004 still training on 6,7
if [[ -f /root/logs/r1004_train.pid ]]; then
  r1004=$(cat /root/logs/r1004_train.pid)
  if kill -0 "$r1004" 2>/dev/null; then
    echo "[p4143] R1004 TRAIN still alive pid=$r1004 (keep)"
  else
    echo "[p4143] WARN R1004 train pid dead"
  fi
fi

bash /root/mining_src/r1011-vera-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-midlr/lean_train_r337_gpus45_p4143.sh

# arm MERGE→n80 waiter now (lean_train arms train→merge; this arms merge→n80)
nohup bash /root/mining_src/r1011-vera-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-midlr/wait_r1011_merge_then_n80_p4143.sh \
  >/root/logs/p4143_r1011_merge_then_n80.outer.nohup 2>&1 &
echo $! >/root/logs/p4143_r1011_merge_then_n80.outer.pid

echo "[p4143] R1011 pid=$(cat /root/logs/r1011_train.pid 2>/dev/null) merge_n80_outer=$(cat /root/logs/p4143_r1011_merge_then_n80.outer.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4143_r997_refute_r1011_armed.done
echo "[p4143] DONE"
