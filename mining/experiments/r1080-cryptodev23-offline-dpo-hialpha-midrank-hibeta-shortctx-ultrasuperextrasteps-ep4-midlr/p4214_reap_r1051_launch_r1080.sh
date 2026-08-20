#!/usr/bin/env bash
# p4214: R1051 REFUTE → exact-PID reap stale chall :8002 → R1080 ShortCtx Hiβ Ultra MidLR TRAIN on r926 GPUs3,4.
# Never pkill -f. Do not touch teacher :8000 (TP4 on 0,1,5,6) or king :8001 (GPU2).
set -euo pipefail
LOG=/root/logs/p4214_reap_r1051_launch_r1080.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4214-r926] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4214-r926] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1051.pid /root/logs/r1051_sim_wvk7.pid \
  /root/logs/r1051_merge_then_n80.pid /root/logs/r1051_wait_merge.pid \
  /root/logs/vllm_chall_r1060.pid /root/logs/r1060_sim_wvk7.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

stop_pid 142480 "chall r1051 :8002"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1051 chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r1051_merged/ && !/awk/ {print $1}')

python3 - <<'PY'
import os, signal, subprocess, time
want={3,4}
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
protect=set()
for line in subprocess.check_output(["ps","-eo","pid=,args="], text=True).splitlines():
  parts=line.strip().split(None,1)
  if len(parts)<2: continue
  pid=int(parts[0]); cmd=parts[1]
  if "GLM-4.5-Air" in cmd or ":8000" in cmd or ("vera6" in cmd and "--port 8001" in cmd):
    protect.add(pid)
kill -= protect
print(f"[p4214-r926] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4214-r926] GPUs 3,4 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4214-r926] wait free 3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; nvidia-smi; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4214-r926] TK warm"

BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
test -e "$BASE/config.json" || { echo FATAL missing cryptoDev base; exit 1; }

chmod +x /root/mining_src/r1080-cryptodev23-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr/*.sh
nohup bash /root/mining_src/r1080-cryptodev23-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr/lean_train_h100_gpus34_p4214.sh >/root/logs/p4214_r1080_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4214_r1080_lean_outer.pid
sleep 12
echo "[p4214-r926] r1080 train.pid=$(cat /root/logs/r1080_train.pid 2>/dev/null || echo MISSING)"
tail -40 /root/logs/r1080_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4214_r1051_refute_r1080_armed.done
echo "[p4214-r926] DONE"
