#!/usr/bin/env bash
# p4165: R1015 v4 REFUTE → exact-PID reap chall:8003 → R1032 ShortCtx MidRank Hiβ Ultra MidLR TRAIN on r924 1,3.
set -euo pipefail
LOG=/root/logs/p4165_reap_r1015_launch_r1032.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4165] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4165] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1015.pid /root/logs/r1015_sim_wvk7.pid \
  /root/logs/r1015_merge_then_n80.pid /root/logs/p4149_r1015_lean_outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
# known stale chall from harvest
stop_pid 80223 "chall r1015 :8003 known"
stop_pid 80885 "r1015 worker_tp0"
stop_pid 80886 "r1015 worker_tp1"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8003"
done < <(ss -lptn "sport = :8003" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1015 chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r1015_merged/ && !/awk/ {print $1}')

python3 - <<'PY'
import os, signal, subprocess, time
want={1,3}
out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"], text=True)
uu={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  idx,u=line.split(","); uu[u.strip()]=int(idx.strip())
kill=set()
for line in out.splitlines():
  parts=[p.strip() for p in line.split(",")]
  if len(parts)<2: continue
  pid=int(parts[0]); u=parts[1]
  gi=uu.get(u)
  if gi in want: kill.add(pid)
print(f"[p4165] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4165] GPUs 1,3 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4165] wait free 1+3 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
# Do NOT touch R1016 on 4,5 or R1026 on 6,7
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4165] TK still warm; R1016/R1026 left alone"

EXP=r1032-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r924_gpus13_p4165.sh
chmod +x /root/mining_src/$EXP/*.sh

nohup bash /root/mining_src/$EXP/lean_train_r924_gpus13_p4165.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4165_r1032_lean_outer.pid
echo "[p4165] R1032 lean launched outer_pid=$(cat /root/logs/p4165_r1032_lean_outer.pid)"

sleep 25
echo "=== r1032 warm ==="; tail -60 /root/logs/r1032_lean_warm.log || true
ps -p "$(cat /root/logs/r1032_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4165_r1015_refute_r1032_armed.done
echo "[p4165] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
