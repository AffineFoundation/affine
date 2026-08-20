#!/usr/bin/env bash
# p4141: R994+R995 REFUTE → exact-PID reap chall :8002/:8003 → R1009+R1010 MidLR TRAIN + MERGE→n80 waiters
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1006 TRAIN 1,3.
set -euo pipefail
LOG=/root/logs/p4141_crown_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4141] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R994+R995 → R1009+R1010"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4141] kill pid=$pid ($why)"
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
  /root/logs/r994_sim_wvk7.pid \
  /root/logs/r995_sim_wvk7.pid \
  /root/logs/p4137_r994_merge_then_n80.outer.pid \
  /root/logs/p4137_r995_merge_then_n80.outer.pid \
  /root/logs/p4137_r994_merge_then_n80.pid \
  /root/logs/p4137_r995_merge_then_n80.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

stop_pidfile /root/logs/vllm_chall_r994.pid "r994 chall"
stop_pidfile /root/logs/vllm_chall_r995.pid "r995 chall"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8003"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r994_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r994_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r995_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r995_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r994"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_r994/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r995"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_r995/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "run_sim_duel r994/r995"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*(r994|r995)/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4141] wait free GPUs u67=$u67 u45=$u45 iter=$i"
  [[ "$u67" -lt 8192 && "$u45" -lt 8192 ]] && break
  sleep 2
done
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$u67" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy u67=$u67"; exit 1; }
[[ "$u45" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy u45=$u45"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4141] TK still warm"

# Keep R1006 train if alive
if [[ -f /root/logs/r1006_train.pid ]]; then
  tp=$(cat /root/logs/r1006_train.pid)
  if kill -0 "$tp" 2>/dev/null; then
    echo "[p4141] R1006 TRAIN still alive pid=$tp (keep)"
  fi
fi

bash /root/mining_src/r1009-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-midlr/lean_train_crown_gpus67_p4141.sh
bash /root/mining_src/r1010-vera-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-midlr/lean_train_crown_gpus45_p4141.sh

echo "[p4141] R1009 pid=$(cat /root/logs/r1009_train.pid 2>/dev/null)"
echo "[p4141] R1010 pid=$(cat /root/logs/r1010_train.pid 2>/dev/null)"

nohup bash /root/mining_src/r1009-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-midlr/wait_r1009_merge_then_n80_p4141.sh \
  >/root/logs/p4141_r1009_merge_then_n80.outer.nohup 2>&1 &
echo $! | tee /root/logs/p4141_r1009_merge_then_n80.pid
nohup bash /root/mining_src/r1010-vera-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-midlr/wait_r1010_merge_then_n80_p4141.sh \
  >/root/logs/p4141_r1010_merge_then_n80.outer.nohup 2>&1 &
echo $! | tee /root/logs/p4141_r1010_merge_then_n80.pid
echo "[p4141] MERGE→n80 waiters r1009=$(cat /root/logs/p4141_r1009_merge_then_n80.pid) r1010=$(cat /root/logs/p4141_r1010_merge_then_n80.pid)"

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4141_r994_r995_refute_r1009_r1010_armed.done
echo "[p4141] DONE"
