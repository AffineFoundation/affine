#!/usr/bin/env bash
# p4133: R989 REFUTE → exact-PID reap chall :8002 → R1004 TRAIN on GPUs 6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R997 TRAIN 4,5.
set -euo pipefail
LOG=/root/logs/p4133_r337_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4133] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R989 → R1004"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4133] kill pid=$pid ($why)"
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
  /root/logs/r989_sim_wvk7.pid \
  /root/logs/p4132_r989_lean_outer.pid \
  /root/logs/p4132_r989_outer.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

stop_pidfile /root/logs/vllm_chall_r989.pid "r989 chall"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r989_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r989_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r989"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r337_r989/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4133] wait free GPUs u67=$u67 iter=$i"
  [[ "$u67" -lt 8192 ]] && break
  sleep 2
done
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u67" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy u67=$u67"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4133] TK still warm"

# Confirm R997 still training on 4,5
if [[ -f /root/logs/r997_train.pid ]]; then
  r997=$(cat /root/logs/r997_train.pid)
  if kill -0 "$r997" 2>/dev/null; then
    echo "[p4133] R997 TRAIN still alive pid=$r997 (keep)"
  else
    echo "[p4133] WARN R997 train pid dead"
  fi
fi

bash /root/mining_src/r1004-vera-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_r337_gpus67_p4133.sh

echo "[p4133] R1004 pid=$(cat /root/logs/r1004_train.pid 2>/dev/null)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4133_r989_refute_r1004_armed.done
echo "[p4133] DONE"
