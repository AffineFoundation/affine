#!/usr/bin/env bash
# p4136: R986 REFUTE → exact-PID reap chall :8002 → R1005 TRAIN on GPUs 6,7 + MERGE→n80 waiter
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1001 TRAIN 1,3 / R1003 TRAIN 4,5.
set -euo pipefail
LOG=/root/logs/p4136_r924_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4136] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R986 → R1005"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4136] kill pid=$pid ($why)"
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
  /root/logs/r986_sim_wvk7.pid \
  /root/logs/p4131_r986_lean_outer.pid \
  /root/logs/p4131_r986_outer.pid \
  /root/logs/p4131_r986_merge_then_n80.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

stop_pidfile /root/logs/vllm_chall_r986.pid "r986 chall"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r986_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r986_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r986"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r924_gpus67_p4131/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "wait_r986_merge_then_n80"
done < <(ps -eo pid=,args= | awk '/wait_r986_merge_then_n80/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4136] wait free GPUs u67=$u67 iter=$i"
  [[ "$u67" -lt 8192 ]] && break
  sleep 2
done
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u67" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy u67=$u67"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4136] TK still warm"

for tag in r1001 r1003; do
  if [[ -f /root/logs/${tag}_train.pid ]]; then
    tp=$(cat /root/logs/${tag}_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo "[p4136] ${tag} TRAIN still alive pid=$tp (keep)"
    else
      echo "[p4136] WARN ${tag} train pid dead"
    fi
  fi
done

bash /root/mining_src/r1005-vera-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_train_r924_gpus67_p4136.sh

echo "[p4136] R1005 pid=$(cat /root/logs/r1005_train.pid 2>/dev/null)"

# Arm MERGE→n80 waiter immediately (train→merge alone is a known idle failure mode)
nohup bash /root/mining_src/r1005-vera-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr/wait_r1005_merge_then_n80_p4136.sh \
  >/root/logs/p4136_r1005_merge_then_n80.outer.nohup 2>&1 &
echo $! | tee /root/logs/p4136_r1005_merge_then_n80.pid
echo "[p4136] MERGE→n80 waiter pid=$(cat /root/logs/p4136_r1005_merge_then_n80.pid)"

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4136_r986_refute_r1005_armed.done
echo "[p4136] DONE"
