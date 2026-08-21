#!/usr/bin/env bash
# p4232 host-side: R1092 REFUTE → reap crown :8002 GPUs6,7 → R1101 Hyper HiLR TRAIN
set -euo pipefail
EXP=r1101-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1101_p4232.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_crown_gpus67_p4232.sh \
  wait_r1101_train_then_merge_p4232.sh \
  wait_r1101_merge_then_n80_p4232.sh \
  lean_chall_n80_crown_r1101_gpus67_p4232.sh
SSH_HOST=95.133.252.28; SSH_PORT=40298
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/.ralph/known_hosts -o ConnectTimeout=45 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1101_p4232.tgz "root@${SSH_HOST}:/tmp/r1101_p4232.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1101-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1101 /root/affine_data
tar -xzf /tmp/r1101_p4232.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1101/dpo_duel_reason.jsonl
python3 - <<'PY'
import os, signal, subprocess, time, re
def stop(pid, why):
    if not pid: return
    try: os.kill(pid, 0)
    except ProcessLookupError: return
    print(f"stop {pid} ({why})", flush=True)
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: return
    for _ in range(20):
        try: os.kill(pid, 0); time.sleep(0.5)
        except ProcessLookupError: return
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
for pf in [
    "/root/logs/vllm_chall_r1092.pid","/root/logs/vllm_chall_r1092_p4223.pid",
    "/root/logs/r1092_sim_wvk7.pid","/root/logs/r1092_merge_then_n80.pid",
    "/root/logs/r1092_wait_merge.pid","/root/logs/p4223_r1092_lean_outer.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass
try:
    out = subprocess.check_output(["ss","-lptn","sport = :8002"], text=True, stderr=subprocess.DEVNULL)
except Exception: out=""
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
    stop(pid, ":8002")
ps = subprocess.check_output(["ps","-eo","pid=,args="], text=True)
for line in ps.splitlines():
    line=line.strip()
    if not line: continue
    parts=line.split(None,1)
    if len(parts)<2: continue
    pid=int(parts[0]); cmd=parts[1]
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "GLM-4.5-Air" in cmd and "8000" in cmd: continue
    if ":8001" in cmd or "/8001" in cmd: continue
    # do not touch sibling R1091/R1093 n80s
    if any(tok in cmd for tok in ["r1091","r1093","8003","8004"]):
        if "r1092" not in cmd: continue
    if any(tok in cmd for tok in [
        "r1092_merged","r1092_sim","local-r1092","vllm_chall_r1092",
        "wait_r1092","lean_chall_n80_crown_r1092","chall_r1092",
        "p4223_r1092",
    ]):
        if "r1101" in cmd: continue
        stop(pid, "argv")
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]; idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (6,7) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: print(f"SKIP train {pid}"); continue
    if "r1101" in cmd or "r1091" in cmd or "r1093" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1092" not in cmd: print(f"SKIP TK {pid}"); continue
    stop(pid, "gpu67")
print("reap done", flush=True)
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
rm -rf /tmp/r1092_merged
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "TK warm"
python3 - <<'PY'
import json,time
from pathlib import Path
Path('/root/affine_data/r1092_refute_p4232.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1092","verdict":"REFUTE",
  "margin":-0.00011503736233372436,"se":0.0019444213037001345,"z":-0.059162776150834255,
  "n":79,"bar":0.003888842607400269,"thought_median":197.0,"b_pass":0.379746835443038,
  "mult":-0.0296,"king":"reign36","note":"p4232 → R1101 HyperExtra HiLR isolate"
}, indent=2)+"\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1092_refute_reaped_p4232.done
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus67_p4232.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4232_r1101_lean_outer.pid
sleep 25
echo "r1101 train.pid=$(cat /root/logs/r1101_train.pid 2>/dev/null || echo MISSING)"
tail -40 /root/logs/r1101_lean_warm.log 2>/dev/null || true
ps -eo pid,etime,cmd | grep -E 'r1101|train_dpo.*r1101' | grep -v grep | head -10
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4232_r1092_refute_r1101_armed.done
echo ARMED
REMOTE
