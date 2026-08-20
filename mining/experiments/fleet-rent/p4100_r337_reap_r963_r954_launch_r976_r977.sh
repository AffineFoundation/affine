#!/usr/bin/env bash
# p4100: R963+R954 REFUTE → exact-PID reap chall :8002/:8003 → R976+R977 TRAIN on GPUs 4,5 / 6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
LOG=/root/logs/p4100_r337_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4100] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R963/R954 → R976/R977"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4100] kill pid=$pid ($why)"
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
  /root/logs/r963_sim_wvk7.pid /root/logs/r954_sim_wvk7.pid \
  /root/logs/p4098_r963_outer.pid /root/logs/p4098_r954_outer.pid \
  /root/logs/p4079_r963_lean_outer.pid /root/logs/p4071_r954_lean_outer.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

# Exact chall pidfiles
stop_pidfile /root/logs/vllm_chall_r963.pid "r963 chall"
stop_pidfile /root/logs/vllm_chall_r954.pid "r954 chall"

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
  stop_pid "$pid" "argv r963/r954_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r(963|954)_merged/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4100] wait free GPUs u45=$u45 u67=$u67 iter=$i"
  [[ "$u45" -lt 8192 && "$u67" -lt 8192 ]] && break
  sleep 2
done
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u45" -lt 8192 && "$u67" -lt 8192 ]] || { echo "FATAL GPUs still busy u45=$u45 u67=$u67"; exit 1; }

# Confirm TK still up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4100] TK still warm"

bash /root/mining_src/r976-vera-offline-dpo-hialpha-midrank-lobeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_train_r337_gpus45_p4100.sh
bash /root/mining_src/r977-vera-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-hilr/lean_train_r337_gpus67_p4100.sh

echo "[p4100] R976 pid=$(cat /root/logs/r976_train.pid 2>/dev/null) R977 pid=$(cat /root/logs/r977_train.pid 2>/dev/null)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4100_r963_r954_refute_r976_r977_armed.done
echo "[p4100] DONE"
