#!/usr/bin/env bash
# p4103: idle R959 chall:8002 (SUBMITTED) → exact-PID reap → R979 SoftCtx HiRank Midβ Mega HiLR TRAIN on R338 GPUs 4,5.
# Keep T:8000 K:8001 R978 TRAIN 6,7. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4103_reap_r959_chall_launch_r979.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4103] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4103] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# Exact PIDs from pidfiles / listeners — never pkill -f
for pf in /root/logs/vllm_chall_r959.pid /root/logs/r959_sim_wvk7.pid /root/logs/p4099_r959_outer.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

python3 - <<'PY'
import os, signal, subprocess, time
want={4,5}
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
print(f"[p4103] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4103] GPUs 4,5 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4103] wait free 4,5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs still busy; exit 1; }

EXP=r979-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-hilr
test -f /root/mining_src/$EXP/lean_train_r338_gpus45_p4103.sh
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_r338_gpus45_p4103.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4103_r979_lean_outer.pid
echo "[p4103] R979 lean launched outer_pid=$(cat /root/logs/p4103_r979_lean_outer.pid)"
sleep 8
tail -40 /root/logs/r979_lean_warm.log || true
ps -p "$(cat /root/logs/r979_train.pid 2>/dev/null)" -o pid,cmd= || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4103_r959_idle_r979_armed.done
echo "[p4103] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
