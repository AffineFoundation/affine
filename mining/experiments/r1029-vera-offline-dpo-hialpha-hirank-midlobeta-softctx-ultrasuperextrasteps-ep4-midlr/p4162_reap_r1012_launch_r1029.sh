#!/usr/bin/env bash
# p4162: R1012 v4 REFUTE → exact-PID reap chall:8002 → R1029 SoftCtx HiRank MidLoβ Ultra MidLR TRAIN on r938 2,3.
# Never pkill -f. Do not touch TK on 0,1.
set -euo pipefail
LOG=/root/logs/p4162_reap_r1012_launch_r1029.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4162] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4162] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1012.pid /root/logs/r1012_sim_wvk7.pid \
  /root/logs/r1012_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
stop_pid 28699 "chall r1012 :8002 known"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

python3 - <<'PY'
import os, signal, subprocess, time
want={2,3}
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
  if gi not in want: continue
  try:
    cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
  except Exception:
    cmd=""
  if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
    if "r1012_merged" not in cmd and "r1029" not in cmd:
      print(f"[p4162] SKIP TK pid={pid}", flush=True); continue
  kill.add(pid)
print(f"[p4162] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4162] GPUs 2,3 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4162] wait free 2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4162] TK warm"

EXP=r1029-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r938_gpus23_p4162.sh
chmod +x /root/mining_src/$EXP/*.sh

nohup bash /root/mining_src/$EXP/lean_train_r938_gpus23_p4162.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4162_r1029_lean_outer.pid
echo "[p4162] R1029 lean launched outer_pid=$(cat /root/logs/p4162_r1029_lean_outer.pid)"

sleep 25
echo "=== r1029 warm ==="; tail -50 /root/logs/r1029_lean_warm.log || true
ps -p "$(cat /root/logs/r1029_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4162_r1012_refute_r1029_armed.done
echo "[p4162] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
