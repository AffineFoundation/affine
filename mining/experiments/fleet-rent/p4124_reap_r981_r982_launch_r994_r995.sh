#!/usr/bin/env bash
# p4124: R981+R982 v4 REFUTE → exact-PID reap chall :8002/:8003 → R994+R995 Mega UltraLoLR TRAIN on crown 6,7 / 4,5.
# Keep T:8000 K:8001 and R993 TRAIN on 1,3. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4124_reap_r981_r982_launch_r994_r995.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4124] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4124] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r981.pid /root/logs/r981_sim_wvk7.pid /root/logs/p4120_r981_chall_n80.outer.pid \
  /root/logs/vllm_chall_r982.pid /root/logs/r982_sim_wvk7.pid /root/logs/p4120_r982_chall_n80.outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done
# known chall parents from poll
stop_pid 142748 "chall vllm r981 :8002"
stop_pid 142874 "chall vllm r982 :8003"

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
  if gi in want:
    kill.add(pid)
# never touch R993 trainer on 1,3
print(f"[p4124] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4124] GPUs 4,5,6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4124] wait free 4,5,6,7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }

# Keep TK + R993 warm
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
if [[ -f /root/logs/r993_train.pid ]] && kill -0 "$(cat /root/logs/r993_train.pid)" 2>/dev/null; then
  echo "[p4124] R993 TRAIN still alive pid=$(cat /root/logs/r993_train.pid)"
else
  echo "[p4124] WARN R993 train pid missing/dead"
fi
echo "[p4124] TK still warm"

EXP994=r994-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr
EXP995=r995-vera-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
test -f /root/mining_src/$EXP994/lean_train_crown_gpus67_p4124.sh
test -f /root/mining_src/$EXP995/lean_train_crown_gpus45_p4124.sh
chmod +x /root/mining_src/$EXP994/*.sh /root/mining_src/$EXP995/*.sh
nohup bash /root/mining_src/$EXP994/lean_train_crown_gpus67_p4124.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4124_r994_lean_outer.pid
nohup bash /root/mining_src/$EXP995/lean_train_crown_gpus45_p4124.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4124_r995_lean_outer.pid
echo "[p4124] R994 lean outer=$(cat /root/logs/p4124_r994_lean_outer.pid) R995 lean outer=$(cat /root/logs/p4124_r995_lean_outer.pid)"
sleep 15
echo '--- r994 lean ---'; tail -40 /root/logs/r994_lean_warm.log || true
echo '--- r995 lean ---'; tail -40 /root/logs/r995_lean_warm.log || true
ps -p "$(cat /root/logs/r994_train.pid 2>/dev/null || echo 0)" -o pid,cmd= 2>/dev/null || true
ps -p "$(cat /root/logs/r995_train.pid 2>/dev/null || echo 0)" -o pid,cmd= 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4124_r981_r982_refute_r994_r995_armed.done
echo "[p4124] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
