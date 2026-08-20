#!/usr/bin/env bash
# p4122: R980 v4 REFUTE → exact-PID reap chall:8002 → R992 SoftCtx HiRank Midβ Mega MidLR TRAIN on R938 GPUs 2,3.
# Keep T:8000 K:8001. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4122_reap_r980_launch_r992.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4122] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4122] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in /root/logs/vllm_chall_r980.pid /root/logs/r980_sim_wvk7.pid /root/logs/p4120_r980_chall_n80.outer.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
# known chall pid from poll
stop_pid 20858 "chall vllm known"

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
  if gi in want:
    kill.add(pid)
print(f"[p4122] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4122] GPUs 2,3 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4122] wait free 2,3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs still busy; exit 1; }

# Keep TK warm
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4122] TK still warm"

EXP=r992-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r938_gpus23_p4122.sh
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_r938_gpus23_p4122.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4122_r992_lean_outer.pid
echo "[p4122] R992 lean launched outer_pid=$(cat /root/logs/p4122_r992_lean_outer.pid)"
sleep 12
tail -60 /root/logs/r992_lean_warm.log || true
ps -p "$(cat /root/logs/r992_train.pid 2>/dev/null)" -o pid,cmd= || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4122_r980_refute_r992_armed.done
echo "[p4122] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
