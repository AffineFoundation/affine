#!/usr/bin/env bash
# p4161: R1017 v4 REFUTE → exact-PID reap chall:8002 → R1027 SoftCtx MidRank Hiβ Mega MidLR TRAIN on r338 6,7.
# Never pkill -f. Do not touch R1018 on 4,5/:8003 or TK.
set -euo pipefail
LOG=/root/logs/p4161_reap_r1017_launch_r1027.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4161] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4161] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1017.pid /root/logs/r1017_sim_wvk7.pid \
  /root/logs/r1017_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
stop_pid 106706 "chall r1017 :8002 known"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

# Never touch :8003 (R1018)
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
  if gi not in want: continue
  try:
    cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
  except Exception:
    cmd=""
  if "r1018" in cmd or ":8003" in cmd or "r1018_merged" in cmd:
    print(f"[p4161] SKIP r1018 pid={pid}", flush=True); continue
  if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
    if "r1017_merged" not in cmd and "r1027" not in cmd:
      print(f"[p4161] SKIP TK pid={pid}", flush=True); continue
  kill.add(pid)
print(f"[p4161] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4161] GPUs 6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4161] wait free 6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
# Do NOT touch R1018 on 4,5
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null || echo "[p4161] WARN :8003 not up (R1018 may still be loading)"
echo "[p4161] TK warm; R1018 left alone on 4,5/:8003"

EXP=r1027-vera-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r338_gpus67_p4161.sh
chmod +x /root/mining_src/$EXP/*.sh

nohup bash /root/mining_src/$EXP/lean_train_r338_gpus67_p4161.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4161_r1027_lean_outer.pid
echo "[p4161] R1027 lean launched outer_pid=$(cat /root/logs/p4161_r1027_lean_outer.pid)"

sleep 25
echo "=== r1027 warm ==="; tail -50 /root/logs/r1027_lean_warm.log || true
ps -p "$(cat /root/logs/r1027_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4161_r1017_refute_r1027_armed.done
echo "[p4161] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
