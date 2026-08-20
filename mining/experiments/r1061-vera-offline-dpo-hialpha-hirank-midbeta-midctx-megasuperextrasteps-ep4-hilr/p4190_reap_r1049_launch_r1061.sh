#!/usr/bin/env bash
# p4190 host: exact-PID reap R1049 chall on r338 :8003 GPUs4,5 → launch R1061 TRAIN. Never pkill -f.
set -euo pipefail
POD=calm-fox-6a
EXP=r1061-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining

# Upload experiment pack (skip large jsonl if already on pod — still include for safety)
tar -C "$ROOT" -czf /tmp/r1061_p4190.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r338_gpus45_p4190.sh \
  wait_r1061_train_then_merge_p4190.sh \
  wait_r1061_merge_then_n80_p4190.sh \
  lean_chall_n80_r338_gpus45_p4190.sh

lium scp "$POD" /tmp/r1061_p4190.tgz /tmp/r1061_p4190.tgz

lium exec "$POD" -- bash -s <<'REMOTE'
set -euo pipefail
EXP=r1061-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1061
tar -xzf /tmp/r1061_p4190.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1061/dpo_duel_reason.jsonl

# Exact-PID reap: R1049 sim + chall vllm + waiters (NOT train_dpo R1050, NOT TK)
python3 - <<'PY'
import os, signal, subprocess, time, re
def stop(pid, why):
    if not pid: return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    print(f"stop {pid} ({why})", flush=True)
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: return
    for _ in range(20):
        try: os.kill(pid, 0); time.sleep(0.5)
        except ProcessLookupError: return
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass

# pidfiles
for pf in [
  "/root/logs/vllm_chall_r1049.pid",
  "/root/logs/r1049_merge_then_n80.pid",
  "/root/logs/r1049_wait_merge.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass

# :8003 listeners
try:
    out = subprocess.check_output(["ss","-lptn","sport = :8003"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
    stop(pid, ":8003")

# argv match for r1049 chall/sim only
ps = subprocess.check_output(["ps","-eo","pid=,args="], text=True)
for line in ps.splitlines():
    line=line.strip()
    if not line: continue
    parts=line.split(None,1)
    if len(parts)<2: continue
    pid=int(parts[0]); cmd=parts[1]
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "GLM-4.5-Air" in cmd and "8000" in cmd: continue
    if "/8001" in cmd or ":8001" in cmd: continue
    if any(tok in cmd for tok in [
        "r1049_merged","r1049_sim","local-r1049","vllm_chall_r1049",
        "wait_r1049","lean_chall_n80_r338_gpus45_p4180","run_sim_duel.py"
    ]):
        # only kill run_sim if it is the r1049 one
        if "run_sim_duel.py" in cmd and "r1049" not in cmd: continue
        stop(pid, "argv")

# GPU 4,5 apps that are r1049 chall (not R1050 train)
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]
    idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (4,5) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: 
        print(f"SKIP train {pid}", flush=True); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1049" not in cmd: print(f"SKIP TK {pid}", flush=True); continue
    if "r1050" in cmd: print(f"SKIP r1050 {pid}", flush=True); continue
    stop(pid, "gpu45")
print("reap done", flush=True)
PY

# Wait GPUs free
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }

# Mark R1049 REFUTE done artifact
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1049_refute_reaped_p4190.done

nohup bash /root/mining_src/$EXP/lean_train_r338_gpus45_p4190.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4190_r1061_lean_outer.pid
sleep 3
echo "OUTER_PID=$(cat /root/logs/p4190_r1061_lean_outer.pid)"
echo "TRAIN_PID=$(cat /root/logs/r1061_train.pid 2>/dev/null || echo pending)"
tail -n 30 /root/logs/r1061_lean_warm.log 2>/dev/null || true
REMOTE
