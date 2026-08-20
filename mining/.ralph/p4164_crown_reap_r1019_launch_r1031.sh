#!/usr/bin/env bash
# p4164: R1019 v4 REFUTE ~0.95× → exact-PID reap chall:8004 → R1031 Ultra MidLR TRAIN on crown GPUs 1,3.
# Do NOT touch R1021/R1022 merges on GPUs 4-7.
set -euo pipefail
LOG=/root/logs/p4164_reap_r1019_launch_r1031.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4164] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4164] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in /root/logs/vllm_chall_r1019.pid /root/logs/r1019_sim_wvk7.pid /root/logs/p4153_r1019_merge_then_n80.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
stop_pid 185568 "chall r1019 :8004 known"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8004"
done < <(ss -lptn "sport = :8004" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

python3 - <<'PY'
import os, signal, subprocess, time
want={1,3}
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
    try:
      cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception:
      cmd=""
    if "train_dpo" in cmd or "merge_lora" in cmd:
      print(f"[p4164] SKIP train/merge pid={pid}", flush=True); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd:
      print(f"[p4164] SKIP TK pid={pid}", flush=True); continue
    kill.add(pid)
print(f"[p4164] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4164] GPUs 1,3 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4164] wait free 1,3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs still busy; exit 1; }
# Do NOT touch R1021/R1022 on 4-7
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4164] TK still warm; R1021/R1022 left alone on 4-7"

EXP=r1031-vera-offline-dpo-hialpha-lorank-midbeta-midctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_crown_gpus13_p4164.sh
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus13_p4164.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4164_r1031_lean_outer.pid
echo "[p4164] R1031 lean launched outer_pid=$(cat /root/logs/p4164_r1031_lean_outer.pid)"
sleep 25
tail -80 /root/logs/r1031_lean_warm.log || true
ps -p "$(cat /root/logs/r1031_train.pid 2>/dev/null)" -o pid,cmd= || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4164_r1019_refute_r1031_armed.done
echo "[p4164] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
