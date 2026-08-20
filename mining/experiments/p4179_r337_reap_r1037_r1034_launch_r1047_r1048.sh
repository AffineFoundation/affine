#!/usr/bin/env bash
# p4179: R1037+R1034 REFUTE → exact-PID reap chall :8002/:8003 → R1048+R1047 TRAIN on r337.
# Never pkill -f. Leave teacher:8000 / king:8001 alone.
set -euo pipefail
LOG=/root/logs/p4179_r337_reap_launch_r1047_r1048.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4179] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4179] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1037.pid /root/logs/r1037_sim_wvk7.pid \
  /root/logs/vllm_chall_r1034.pid /root/logs/r1034_sim_wvk7.pid \
  /root/logs/vllm_chall_r1047.pid /root/logs/vllm_chall_r1048.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

# known stale chall PIDs from inventory
stop_pid 107569 "chall r1037 :8002"
stop_pid 104444 "chall r1034 :8003"

for port in 8002 8003; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale merged chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r103[47]_merged/ && !/awk/ {print $1}')

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
print(f"[p4179] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4179] GPUs 4-7 reaped", flush=True)
PY

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4179] wait free 4-7 used_mib=$used iter=$i"
  [[ "$used" -lt 32768 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 32768 ]] || { echo FATAL GPUs still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4179] TK warm"

for EXP in \
  r1047-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-hilr \
  r1048-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-midlr
do
  test -d /root/mining_src/$EXP
  chmod +x /root/mining_src/$EXP/*.sh
done

nohup bash /root/mining_src/r1047-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_train_r337_gpus45_p4179.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4179_r1047_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r1048-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-midlr/lean_train_r337_gpus67_p4179.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4179_r1048_lean_outer.pid

sleep 20
echo "=== r1047 warm ==="; tail -40 /root/logs/r1047_lean_warm.log || true
echo "=== r1048 warm ==="; tail -40 /root/logs/r1048_lean_warm.log || true
echo "=== pids ==="; cat /root/logs/r1047_train.pid /root/logs/r1048_train.pid 2>/dev/null || true
ps -p "$(cat /root/logs/r1047_train.pid 2>/dev/null)" -o pid,cmd= 2>/dev/null || true
ps -p "$(cat /root/logs/r1048_train.pid 2>/dev/null)" -o pid,cmd= 2>/dev/null || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4179_r337_r1047_r1048_armed.done
echo "[p4179] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
