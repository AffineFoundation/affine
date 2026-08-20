#!/usr/bin/env bash
# p4180: R1038+R1039 REFUTE → exact-PID reap stale challs → R1049+R1050 TRAIN on r338.
# Never pkill -f. Do not touch teacher :8000 or king :8001.
set -euo pipefail
LOG=/root/logs/p4180_reap_r1038_r1039_launch_r1049_r1050.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4180] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4180] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# stop waiters / sims for resolved axes
for pf in \
  /root/logs/vllm_chall_r1038.pid /root/logs/r1038_sim_wvk7.pid \
  /root/logs/r1038_merge_then_n80.pid /root/logs/r1038_wait_merge.pid \
  /root/logs/vllm_chall_r1039.pid /root/logs/r1039_sim_wvk7.pid \
  /root/logs/r1039_merge_then_n80.pid /root/logs/r1039_wait_merge.pid \
  /root/logs/p4168_r1038_lean_outer.pid /root/logs/p4169_r1039_lean_outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

# known live challs from inventory
stop_pid 126437 "chall r1038 :8003"
stop_pid 129918 "chall r1039 :8002"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8003"
done < <(ss -lptn "sport = :8003" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1038/r1039 chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r103[89]_merged/ && !/awk/ {print $1}')

python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
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
# never kill teacher/king
protect=set()
for line in subprocess.check_output(["ps","-eo","pid=,args="], text=True).splitlines():
  parts=line.strip().split(None,1)
  if len(parts)<2: continue
  pid=int(parts[0]); cmd=parts[1]
  if "GLM-4.5-Air" in cmd or ":8000" in cmd or ("vera6" in cmd and "--port 8001" in cmd):
    protect.add(pid)
kill -= protect
print(f"[p4180] reap gpu={sorted(want)} kill={sorted(kill)} protect={sorted(protect)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4180] GPUs 4-7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4180] wait free 4-7 used_mib=$used iter=$i"
  [[ "$used" -lt 32768 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 32768 ]] || { echo FATAL GPUs still busy; nvidia-smi; exit 1; }

# confirm TK still up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4180] TK warm"

chmod +x /root/mining_src/r1049-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1050-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1049-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-hilr/lean_train_r338_gpus45_p4180.sh >/root/logs/p4180_r1049_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4180_r1049_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r1050-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-hilr/lean_train_r338_gpus67_p4180.sh >/root/logs/p4180_r1050_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4180_r1050_lean_outer.pid

sleep 8
echo "[p4180] r1049 train.pid=$(cat /root/logs/r1049_train.pid 2>/dev/null || echo MISSING)"
echo "[p4180] r1050 train.pid=$(cat /root/logs/r1050_train.pid 2>/dev/null || echo MISSING)"
tail -20 /root/logs/r1049_lean_warm.log 2>/dev/null || true
tail -20 /root/logs/r1050_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4180_r1038_r1039_refute_r1049_r1050_armed.done
echo "[p4180] DONE"
