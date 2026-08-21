#!/usr/bin/env bash
# p4231 host-side: R1087 REFUTE → reap r338 :8003 GPUs4,5 → R1100 Hyper HiLR TRAIN
set -euo pipefail
EXP=r1100-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1100_p4231.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r338_gpus45_p4231.sh \
  wait_r1100_train_then_merge_p4231.sh \
  wait_r1100_merge_then_n80_p4231.sh \
  lean_chall_n80_r338_gpus45_p4231.sh
SSH_HOST=95.133.253.90; SSH_PORT=40099
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1100_p4231.tgz "root@${SSH_HOST}:/tmp/r1100_p4231.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1100-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1100 /root/affine_data
tar -xzf /tmp/r1100_p4231.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1100/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1087.pid","/root/logs/vllm_chall_r1087_p4220.pid",
    "/root/logs/r1087_sim_wvk7.pid","/root/logs/r1087_merge_then_n80.pid",
    "/root/logs/r1087_wait_merge.pid","/root/logs/p4220_r1087_lean_outer.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass
try:
    out = subprocess.check_output(["ss","-lptn","sport = :8003"], text=True, stderr=subprocess.DEVNULL)
except Exception: out=""
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
    stop(pid, ":8003")
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
    if any(tok in cmd for tok in [
        "r1087_merged","r1087_sim","local-r1087","vllm_chall_r1087",
        "wait_r1087","lean_chall_n80_r338_gpus45_p4220","chall_r1087",
        "p4220_r1087",
    ]):
        if "r1100" in cmd or "r1088" in cmd: continue
        stop(pid, "argv")
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]; idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (4,5) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: print(f"SKIP train {pid}"); continue
    if "r1100" in cmd or "r1088" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1087" not in cmd: print(f"SKIP TK {pid}"); continue
    stop(pid, "gpu45")
print("reap done", flush=True)
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
rm -rf /tmp/r1087_merged
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "TK warm"
python3 - <<'PY'
import json,time
from pathlib import Path
Path('/root/affine_data/r1087_refute_p4231.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1087","verdict":"REFUTE",
  "margin":-0.0022179098466830207,"se":0.00253159456569286,"z":-0.8760920396730317,
  "n":80,"bar":0.00506318913138572,"thought_median":198.0,"b_pass":0.42677824267782427,
  "mult":-0.438,"king":"reign36","note":"p4231 → R1100 HyperExtra HiLR isolate"
}, indent=2)+"\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1087_refute_reaped_p4231.done
nohup bash /root/mining_src/$EXP/lean_train_r338_gpus45_p4231.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4231_r1100_lean_outer.pid
sleep 25
echo "r1100 train.pid=$(cat /root/logs/r1100_train.pid 2>/dev/null || echo MISSING)"
tail -50 /root/logs/r1100_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4231_r1087_refute_r1100_armed.done
echo DONE
REMOTE
