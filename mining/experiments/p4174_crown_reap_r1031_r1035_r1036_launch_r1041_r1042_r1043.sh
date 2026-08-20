#!/usr/bin/env bash
# p4174: R1031+R1035+R1036 v4 REFUTE → exact-PID reap stale challs → R1041/R1042/R1043 TRAIN on crown.
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
LOG=/root/logs/p4174_crown_reap_launch_r1041_r1042_r1043.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4174] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4174] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# known stale chall PIDs from p4174 harvest
stop_pid 199968 "chall r1031 :8004 known"
stop_pid 203305 "chall r1035 :8002 known"
stop_pid 206984 "chall r1036 :8003 known"

for pf in \
  /root/logs/vllm_chall_r1031.pid /root/logs/vllm_chall_r1035.pid /root/logs/vllm_chall_r1036.pid \
  /root/logs/r1031_sim_wvk7.pid /root/logs/r1035_sim_wvk7.pid /root/logs/r1036_sim_wvk7.pid \
  /root/logs/r1031_merge_then_n80.pid /root/logs/r1035_merge_then_n80.pid /root/logs/r1036_merge_then_n80.pid \
  /root/logs/p4164_r1031_merge_then_n80.pid /root/logs/p4166_r1035_merge_then_n80.pid /root/logs/p4166_r1036_merge_then_n80.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

for port in 8002 8003 8004; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done

for tag in r1031_merged r1035_merged r1036_merged; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "stale $tag argv"
  done < <(ps -eo pid=,args= | awk -v t="$tag" '$0 ~ ("vllm serve .*/tmp/" t) && !/awk/ {print $1}')
done

python3 - <<'PY'
import os, signal, subprocess, time
want={1,3,4,5,6,7}
# Protect teacher GPU0 + king GPU2
out=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"], text=True)
uu={}
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True).splitlines():
  idx,u=line.split(","); uu[u.strip()]=int(idx.strip())
# Never kill teacher/king PIDs
protect=set()
for line in subprocess.check_output(["ps","-eo","pid=,args="], text=True).splitlines():
  if "vllm serve" in line and (":8000" in line or "GLM-4.5-Air-FP8" in line or ":8001" in line or "vera6" in line and "affine-5g4yy75zuz-t6" in line):
    protect.add(int(line.split()[0]))
kill=set()
for line in out.splitlines():
  parts=[p.strip() for p in line.split(",")]
  if len(parts)<2: continue
  pid=int(parts[0]); u=parts[1]
  gi=uu.get(u)
  if gi in want and pid not in protect:
    kill.add(pid)
print(f"[p4174] reap gpu={sorted(want)} protect={sorted(protect)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4174] GPUs 1,3,4,5,6,7 reaped", flush=True)
PY

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4174] wait free 1,3,4,5,6,7 used_mib=$used iter=$i"
  [[ "$used" -lt 49152 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 49152 ]] || { echo FATAL GPUs still busy used=$used; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4174] TK still warm"

for EXP in \
  r1041-vera-offline-dpo-hialpha-lorank-midbeta-midctx-megasuperextrasteps-ep4-hilr \
  r1042-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr \
  r1043-vera-offline-dpo-hialpha-midrank-lobeta-softctx-ultrasuperextrasteps-ep4-hilr
do
  test -d /root/mining_src/$EXP
  chmod +x /root/mining_src/$EXP/*.sh
done

nohup bash /root/mining_src/r1041-vera-offline-dpo-hialpha-lorank-midbeta-midctx-megasuperextrasteps-ep4-hilr/lean_train_crown_gpus13_p4174.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4174_r1041_lean_outer.pid
echo "[p4174] R1041 lean outer_pid=$(cat /root/logs/p4174_r1041_lean_outer.pid)"

nohup bash /root/mining_src/r1042-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr/lean_train_crown_gpus67_p4174.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4174_r1042_lean_outer.pid
echo "[p4174] R1042 lean outer_pid=$(cat /root/logs/p4174_r1042_lean_outer.pid)"

nohup bash /root/mining_src/r1043-vera-offline-dpo-hialpha-midrank-lobeta-softctx-ultrasuperextrasteps-ep4-hilr/lean_train_crown_gpus45_p4174.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4174_r1043_lean_outer.pid
echo "[p4174] R1043 lean outer_pid=$(cat /root/logs/p4174_r1043_lean_outer.pid)"

sleep 30
for rid in r1041 r1042 r1043; do
  echo "=== $rid warm ==="; tail -40 /root/logs/${rid}_lean_warm.log || true
  ps -p "$(cat /root/logs/${rid}_train.pid 2>/dev/null)" -o pid,cmd= || true
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4174_crown_r1041_r1042_r1043_armed.done
echo "[p4174] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
