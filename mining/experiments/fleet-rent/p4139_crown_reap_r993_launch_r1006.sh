#!/usr/bin/env bash
# p4139: R993 REFUTE → exact-PID reap chall :8004 → R1006 TRAIN on GPUs 1,3 + MERGE→n80 waiter
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R994 TRAIN/n80 6,7 / R995 TRAIN/n80 4,5.
set -euo pipefail
LOG=/root/logs/p4139_crown_reap_launch.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4139] $(date -u +%Y-%m-%dT%H:%M:%SZ) START reap R993 → R1006"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4139] kill pid=$pid ($why)"
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
  /root/logs/r993_sim_wvk7.pid \
  /root/logs/p4137_r993_merge_then_n80.outer.pid \
  /root/logs/p4137_r993_merge_then_n80.pid
do
  stop_pidfile "$pf" "stale outer/sim"
done

stop_pidfile /root/logs/vllm_chall_r993.pid "r993 chall"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "port :8004"
done < <(ss -lptn 'sport = :8004' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "argv r993_merged"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r993_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "lean_chall r993"
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_crown_r993_gpus13/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "wait_r993_merge_then_n80"
done < <(ps -eo pid=,args= | awk '/wait_r993_merge_then_n80/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "run_sim_duel r993"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r993/ && !/awk/ {print $1}')

for i in $(seq 1 90); do
  u13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4139] wait free GPUs u13=$u13 iter=$i"
  [[ "$u13" -lt 8192 ]] && break
  sleep 2
done
u13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$u13" -lt 8192 ]] || { echo "FATAL GPUs 1,3 still busy u13=$u13"; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4139] TK still warm"

for tag in r994 r995; do
  if [[ -f /root/logs/${tag}_train.pid ]]; then
    tp=$(cat /root/logs/${tag}_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo "[p4139] ${tag} TRAIN still alive pid=$tp (keep)"
    else
      echo "[p4139] ${tag} train pid dead/absent (ok if already n80)"
    fi
  fi
  # keep n80 sims if running
  if [[ -f /root/logs/${tag}_sim_wvk7.pid ]]; then
    sp=$(cat /root/logs/${tag}_sim_wvk7.pid)
    if kill -0 "$sp" 2>/dev/null; then
      echo "[p4139] ${tag} n80 sim alive pid=$sp (keep)"
    fi
  fi
done

bash /root/mining_src/r1006-vera-offline-dpo-hialpha-lorank-midbeta-softctx-megasuperextrasteps-ep4-midlr/lean_train_crown_gpus13_p4139.sh

echo "[p4139] R1006 pid=$(cat /root/logs/r1006_train.pid 2>/dev/null)"

nohup bash /root/mining_src/r1006-vera-offline-dpo-hialpha-lorank-midbeta-softctx-megasuperextrasteps-ep4-midlr/wait_r1006_merge_then_n80_p4139.sh \
  >/root/logs/p4139_r1006_merge_then_n80.outer.nohup 2>&1 &
echo $! | tee /root/logs/p4139_r1006_merge_then_n80.pid
echo "[p4139] MERGE→n80 waiter pid=$(cat /root/logs/p4139_r1006_merge_then_n80.pid)"

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4139_r993_refute_r1006_armed.done
echo "[p4139] DONE"
