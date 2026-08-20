#!/usr/bin/env bash
# p4168: R1028 v4 REFUTE → exact-PID reap chall:8003 → R1038 MidCtx HiRank MidLoβ Mega UltraLoLR TRAIN on r338 4,5.
# Do NOT touch TK (:8000/:8001) or R1027 merge/n80 path on GPUs 6,7. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4168_reap_r1028_launch_r1038.log
mkdir -p /root/logs /root/affine_data
exec > >(tee -a "$LOG") 2>&1
echo "[p4168] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4168] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in \
  /root/logs/vllm_chall_r1028.pid /root/logs/r1028_sim_wvk7.pid \
  /root/logs/r1028_merge_then_n80.pid /root/logs/r1028_lean_outer.pid \
  /root/logs/p4162_r1028_lean_outer.pid
do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)" "$pf"
done

# known chall pid from live ps if still up
stop_pid 116587 "chall r1028 :8003 known"

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "listener :8003"
done < <(ss -lptn "sport = :8003" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1028 chall argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r1028_merged/ && !/awk/ {print $1}')

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "stale r1028 sim"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r1028/ && !/awk/ {print $1}')

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
  if gi in want: kill.add(pid)
print(f"[p4168] reap gpu={sorted(want)} kill={sorted(kill)}", flush=True)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGTERM)
  except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
  try: os.kill(pid, signal.SIGKILL)
  except ProcessLookupError: pass
print("[p4168] GPUs 4,5 reaped", flush=True)
PY

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4168] wait free 4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs still busy; exit 1; }

# Do NOT touch R1027 on 6,7 / merge pid
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4168] TK still warm; R1027 left alone on 6,7"

EXP=r1038-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
test -f /root/mining_src/$EXP/lean_train_r338_gpus45_p4168.sh
chmod +x /root/mining_src/$EXP/*.sh

# copy data if needed
mkdir -p /root/r1038
if [[ ! -s /root/r1038/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1038/dpo_duel_reason.jsonl
  elif [[ -s /root/r1028/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1028/dpo_duel_reason.jsonl /root/r1038/dpo_duel_reason.jsonl
  elif [[ -s /root/r1008/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1008/dpo_duel_reason.jsonl /root/r1038/dpo_duel_reason.jsonl
  else
    echo FATAL no dpo data; exit 1
  fi
fi

nohup bash /root/mining_src/$EXP/lean_train_r338_gpus45_p4168.sh >/root/logs/p4168_r1038_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4168_r1038_lean_outer.pid
sleep 3
TP=$(cat /root/logs/r1038_train.pid 2>/dev/null || true)
echo "[p4168] R1038 TRAIN pid=${TP:-pending} outer=$(cat /root/logs/p4168_r1038_lean_outer.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4168_r1028_refute_r1038_armed.done
echo "[p4168] DONE"
