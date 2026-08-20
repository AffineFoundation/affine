#!/usr/bin/env bash
# p4160: R1005 v4 REFUTE → exact-PID reap chall:8002 → R1026 SoftCtx HiRank Hiβ Mega MidLR TRAIN on r924 6,7.
set -euo pipefail
LOG=/root/logs/p4160_reap_r1005_launch_r1026.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4160] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4160] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1005.pid /root/logs/r1005_sim_wvk7.pid \
  /root/logs/p4136_r1005_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
stop_pid 76209 "chall r1005 :8002 known"
stop_pid 77966 "sim r1005 known"
stop_pid 76106 "lean r1005 bash"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

python3 - <<'PY'
import os, signal, subprocess, time
want={6,7}
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
print(f"[p4160] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4160] GPUs 6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4160] wait free 6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
# Do NOT touch R1015 on 1,3 or R1016 on 4,5
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4160] TK still warm; R1015/R1016 left alone"

EXP=r1026-vera-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r924_gpus67_p4160.sh
chmod +x /root/mining_src/$EXP/*.sh

nohup bash /root/mining_src/$EXP/lean_train_r924_gpus67_p4160.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4160_r1026_lean_outer.pid
echo "[p4160] R1026 lean launched outer_pid=$(cat /root/logs/p4160_r1026_lean_outer.pid)"

sleep 25
echo "=== r1026 warm ==="; tail -50 /root/logs/r1026_lean_warm.log || true
ps -p "$(cat /root/logs/r1026_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4160_r1005_refute_r1026_armed.done
echo "[p4160] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
