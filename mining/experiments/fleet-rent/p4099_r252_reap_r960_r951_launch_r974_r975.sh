#!/usr/bin/env bash
# p4099: R960+R951 REFUTE → exact-PID reap chall :8002/:8003 → R974+R975 TRAIN on GPUs 4,5 / 6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
LOG=/root/logs/p4099_r252_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4099] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R960/R951 → R974/R975"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4099] kill pid=$pid ($why)"
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

# Stop leftover sim/outer wrappers first (exact pidfiles only)
for pf in \
  /root/logs/r960_sim_wvk7.pid /root/logs/r951_sim_wvk7.pid \
  /root/logs/p4097_r960_outer.pid /root/logs/p4097_r951_outer.pid \
  /root/logs/p4074_r960_lean_outer.pid /root/logs/p4065_r951_lean_outer.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

# Exact chall pidfiles
stop_pidfile /root/logs/vllm_chall_r960.pid "r960 chall"
stop_pidfile /root/logs/vllm_chall_r951.pid "r951 chall"

# Port holders on :8002/:8003 only
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

# Any leftover argv-matched chall serve on these merges (not TK)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r960/r951_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r(960|951)_merged/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4099] wait free GPUs u45=$u45 u67=$u67 iter=$i"
  [[ "$u45" -lt 8192 && "$u67" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 && "$u67" -lt 8192 ]] || { echo "FATAL GPUs still busy u45=$u45 u67=$u67"; exit 1; }

# Confirm TK still up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4099] TK still warm"

bash /root/mining_src/r974-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_train_r252_gpus45_p4099.sh
bash /root/mining_src/r975-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-hilr/lean_train_r252_gpus67_p4099.sh

echo "[p4099] R974 pid=$(cat /root/logs/r974_train.pid 2>/dev/null) R975 pid=$(cat /root/logs/r975_train.pid 2>/dev/null)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4099_r960_r951_refute_r974_r975_armed.done
echo "[p4099] DONE"
