#!/usr/bin/env bash
# p4082: R956+R958 REFUTE → exact-PID reap chall:8002/:8004 → R965+R966 HiLR TRAIN on crown GPUs 4,5 / 6,7.
# Never pkill -f. Leave T:8000 K:8001 R957 TRAIN 1,3 alone.
set -euo pipefail
LOG=/root/logs/p4082_reap_r956_r958_launch_r965_r966.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4082] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4082] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r956.pid /root/logs/vllm_chall_r958.pid \
  /root/logs/r956_sim_wvk7.pid /root/logs/r958_sim_wvk7.pid \
  /root/logs/p4080_r956_lean_outer.pid /root/logs/p4080_r958_lean_outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
for port in 8002 8004; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done

python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"], text=True)
uu={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  idx,u=line.split(","); uu[u.strip()]=int(idx.strip())
# Protect teacher/king/R957 train if somehow listed — only kill apps on want GPUs
kill=set()
for line in out.splitlines():
  parts=[p.strip() for p in line.split(",")]
  if len(parts)<2: continue
  pid=int(parts[0]); u=parts[1]
  gi=uu.get(u)
  if gi in want:
    kill.add(pid)
print(f"[p4082] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4082] GPUs 4-7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4082] wait free 4-7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }

test -f /root/mining_src/r965-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-hilr/lean_train_crown_gpus45_p4082.sh
test -f /root/mining_src/r966-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_train_crown_gpus67_p4082.sh

nohup bash /root/mining_src/r965-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-hilr/lean_train_crown_gpus45_p4082.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4082_r965_lean_outer.pid
nohup bash /root/mining_src/r966-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_train_crown_gpus67_p4082.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4082_r966_lean_outer.pid
echo "[p4082] R965 outer=$(cat /root/logs/p4082_r965_lean_outer.pid) R966 outer=$(cat /root/logs/p4082_r966_lean_outer.pid)"
sleep 8
echo "=== r965 warm ==="; tail -30 /root/logs/r965_lean_warm.log || true
echo "=== r966 warm ==="; tail -30 /root/logs/r966_lean_warm.log || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4082_r956_r958_refute_r965_r966_armed.done
echo "[p4082] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
