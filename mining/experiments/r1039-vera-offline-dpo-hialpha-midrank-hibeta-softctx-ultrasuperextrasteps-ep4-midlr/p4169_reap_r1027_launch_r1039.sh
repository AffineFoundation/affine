#!/usr/bin/env bash
# p4169: R1027 v4 REFUTE → exact-PID reap chall:8002 → R1039 SoftCtx MidRank Hiβ Ultra MidLR TRAIN on r338 6,7.
# Do NOT touch TK (:8000/:8001) or R1038 train on GPUs 4,5. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4169_reap_r1027_launch_r1039.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4169] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4169] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1027.pid /root/logs/r1027_sim_wvk7.pid \
  /root/logs/r1027_merge_then_n80.pid /root/logs/r1027_lean_outer.pid \
  /root/logs/p4161_r1027_lean_outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

stop_pid 120736 "chall r1027 :8002 known"
stop_pid 122995 "sim r1027 known"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8002"
done < <(ss -lptn "sport = :8002" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1027 chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r1027_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1027 sim"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r1027/ && !/awk/ {print $1}')

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
  if gi in want: kill.add(pid)
print(f"[p4169] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4169] GPUs 6,7 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4169] wait free 6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4169] TK still warm; R1038 left alone on 4,5"

EXP=r1039-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-midlr
test -f /root/mining_src/$EXP/lean_train_r338_gpus67_p4169.sh
chmod +x /root/mining_src/$EXP/*.sh

mkdir -p /root/r1039
if [[ ! -s /root/r1039/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1039/dpo_duel_reason.jsonl
  elif [[ -s /root/r1027/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1027/dpo_duel_reason.jsonl /root/r1039/dpo_duel_reason.jsonl
  elif [[ -s /root/r978/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r978/dpo_duel_reason.jsonl /root/r1039/dpo_duel_reason.jsonl
  else
    echo FATAL no dpo data; exit 1
  fi
fi

nohup bash /root/mining_src/$EXP/lean_train_r338_gpus67_p4169.sh >/root/logs/p4169_r1039_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4169_r1039_lean_outer.pid
sleep 3
TP=$(cat /root/logs/r1039_train.pid 2>/dev/null || true)
echo "[p4169] R1039 TRAIN pid=${TP:-pending} outer=$(cat /root/logs/p4169_r1039_lean_outer.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4169_r1027_refute_r1039_armed.done
echo "[p4169] DONE"
