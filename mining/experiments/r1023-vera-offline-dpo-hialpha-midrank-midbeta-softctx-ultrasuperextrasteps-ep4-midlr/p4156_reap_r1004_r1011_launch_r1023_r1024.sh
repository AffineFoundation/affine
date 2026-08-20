#!/usr/bin/env bash
# p4156: R1004+R1011 v4 REFUTE → exact-PID reap chall:8002/:8003 → R1023+R1024 TRAIN on r337 6,7 + 4,5.
set -euo pipefail
LOG=/root/logs/p4156_reap_r1004_r1011_launch_r1023_r1024.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4156] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4156] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1004.pid /root/logs/r1004_sim_wvk7.pid \
  /root/logs/vllm_chall_r1011.pid /root/logs/r1011_sim_wvk7.pid \
  /root/logs/p4133_r1004_merge_then_n80.pid /root/logs/p4143_r1011_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
stop_pid 86891 "chall r1004 :8002 known"
stop_pid 89832 "chall r1011 :8003 known"
stop_pid 88722 "sim r1004 known"
stop_pid 91677 "sim r1011 known"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8003"
done < <(ss -lptn "sport = :8003" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

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
print(f"[p4156] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4156] GPUs 4,5,6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4156] wait free 4-7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4156] TK still warm"

EXP23=r1023-vera-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-midlr
EXP24=r1024-vera-offline-dpo-hialpha-midrank-lobeta-midctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP23/lean_train_r337_gpus67_p4156.sh
test -f /root/mining_src/$EXP24/lean_train_r337_gpus45_p4156.sh
chmod +x /root/mining_src/$EXP23/*.sh /root/mining_src/$EXP24/*.sh

nohup bash /root/mining_src/$EXP23/lean_train_r337_gpus67_p4156.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4156_r1023_lean_outer.pid
echo "[p4156] R1023 lean launched outer_pid=$(cat /root/logs/p4156_r1023_lean_outer.pid)"

nohup bash /root/mining_src/$EXP24/lean_train_r337_gpus45_p4156.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4156_r1024_lean_outer.pid
echo "[p4156] R1024 lean launched outer_pid=$(cat /root/logs/p4156_r1024_lean_outer.pid)"

sleep 35
echo "=== r1023 warm ==="; tail -50 /root/logs/r1023_lean_warm.log || true
echo "=== r1024 warm ==="; tail -50 /root/logs/r1024_lean_warm.log || true
ps -p "$(cat /root/logs/r1023_train.pid 2>/dev/null)" -o pid,cmd= || true
ps -p "$(cat /root/logs/r1024_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4156_r1004_r1011_refute_r1023_r1024_armed.done
echo "[p4156] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
