#!/usr/bin/env bash
# p4155: R1009+R1010 v4 REFUTE → exact-PID reap chall:8002/:8003 → R1021+R1022 SoftCtx Ultra MidLR TRAIN on crown 6,7 + 4,5.
set -euo pipefail
LOG=/root/logs/p4155_reap_r1009_r1010_launch_r1021_r1022.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4155] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4155] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1009.pid /root/logs/r1009_sim_wvk7.pid \
  /root/logs/vllm_chall_r1010.pid /root/logs/r1010_sim_wvk7.pid \
  /root/logs/p4141_r1009_merge_then_n80.pid /root/logs/p4141_r1010_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
# known chall vllm pids from harvest
stop_pid 175716 "chall r1009 :8002 known"
stop_pid 176851 "chall r1010 :8003 known"
# also any leftover lean bash parents
stop_pid 176645 "lean r1010 bash"

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
print(f"[p4155] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4155] GPUs 4,5,6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4155] wait free 4-7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }
# Do NOT touch R1019 on 1,3
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4155] TK still warm; R1019 left alone on 1,3"

EXP21=r1021-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-ultrasuperextrasteps-ep4-midlr
EXP22=r1022-vera-offline-dpo-hialpha-midrank-lobeta-softctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP21/lean_train_crown_gpus67_p4155.sh
test -f /root/mining_src/$EXP22/lean_train_crown_gpus45_p4155.sh
chmod +x /root/mining_src/$EXP21/*.sh /root/mining_src/$EXP22/*.sh

nohup bash /root/mining_src/$EXP21/lean_train_crown_gpus67_p4155.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4155_r1021_lean_outer.pid
echo "[p4155] R1021 lean launched outer_pid=$(cat /root/logs/p4155_r1021_lean_outer.pid)"

nohup bash /root/mining_src/$EXP22/lean_train_crown_gpus45_p4155.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4155_r1022_lean_outer.pid
echo "[p4155] R1022 lean launched outer_pid=$(cat /root/logs/p4155_r1022_lean_outer.pid)"

sleep 30
echo "=== r1021 warm ==="; tail -40 /root/logs/r1021_lean_warm.log || true
echo "=== r1022 warm ==="; tail -40 /root/logs/r1022_lean_warm.log || true
ps -p "$(cat /root/logs/r1021_train.pid 2>/dev/null)" -o pid,cmd= || true
ps -p "$(cat /root/logs/r1022_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4155_r1009_r1010_refute_r1021_r1022_armed.done
echo "[p4155] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
