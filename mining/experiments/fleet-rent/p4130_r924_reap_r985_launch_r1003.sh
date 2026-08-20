#!/usr/bin/env bash
# p4130: R985 REFUTE → exact-PID reap chall :8002 → R1003 TRAIN on GPUs 4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1001 TRAIN 1,3 / R986 TRAIN 6,7.
set -euo pipefail
LOG=/root/logs/p4130_r924_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4130-r924] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R985 → R1003"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4130-r924] kill pid=$pid ($why)"
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
  /root/logs/r985_sim_wvk7.pid \
  /root/logs/p4128_r985_lean_outer.pid \
  /root/logs/p4128_r985_outer.pid \
  /root/logs/vllm_chall_r985.pid
do
  stop_pidfile "$pf" "stale outer/sim/chall"
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r985_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r985_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r985"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r924_r985/ && !/awk/ {print $1}')

# Stop orphan engine workers still holding GPUs 4,5 only (never 0–3 / 6–7)
while read -r line; do
  pid=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "gpu4/5 orphan"
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 4,5 2>/dev/null || true)

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4130-r924] wait free GPUs u45=$u45 iter=$i"
  [[ "$u45" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy u45=$u45"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4130-r924] TK still warm"

if [[ -f /root/logs/r1001_train.pid ]]; then
  r1001=$(cat /root/logs/r1001_train.pid)
  if kill -0 "$r1001" 2>/dev/null; then
    echo "[p4130-r924] R1001 TRAIN keep pid=$r1001"
  fi
fi
if [[ -f /root/logs/r986_train.pid ]]; then
  r986=$(cat /root/logs/r986_train.pid)
  if kill -0 "$r986" 2>/dev/null; then
    echo "[p4130-r924] R986 TRAIN keep pid=$r986"
  fi
fi

chmod +x /root/mining_src/r1003-vera-offline-dpo-hialpha-hirank-midbeta-midctx-ultrasuperextrasteps-ep4-ultralolr/*.sh
bash /root/mining_src/r1003-vera-offline-dpo-hialpha-hirank-midbeta-midctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_r924_gpus45_p4130.sh
echo "[p4130-r924] R1003 pid=$(cat /root/logs/r1003_train.pid 2>/dev/null)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4130_r985_refute_r1003_armed.done
echo "[p4130-r924] DONE"
