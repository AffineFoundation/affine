#!/usr/bin/env bash
# p4233 host-side: R1088 REFUTE → reap r338 :8002 GPUs6,7 → R1102 Hyper HiLR TRAIN
set -euo pipefail
EXP=r1102-vera-offline-dpo-hialpha-midrank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1102_p4233.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r338_gpus67_p4233.sh \
  wait_r1102_train_then_merge_p4233.sh \
  wait_r1102_merge_then_n80_p4233.sh \
  lean_chall_n80_r338_gpus67_p4233.sh
SSH_HOST=95.133.253.90; SSH_PORT=40099
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/.ralph/known_hosts -o ConnectTimeout=45 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1102_p4233.tgz "root@${SSH_HOST}:/tmp/r1102_p4233.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1102-vera-offline-dpo-hialpha-midrank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1102 /root/affine_data
tar -xzf /tmp/r1102_p4233.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1102/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1088.pid","/root/logs/vllm_chall_r1088_p4221.pid",
    "/root/logs/r1088_sim_wvk7.pid","/root/logs/r1088_merge_then_n80.pid",
    "/root/logs/r1088_wait_merge.pid","/root/logs/p4221_r1088_lean_outer.pid",
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
    # do not touch sibling R1100 train/n80 on 4,5
    if any(tok in cmd for tok in ["r1100","8003"]):
        if "r1088" not in cmd: continue
    if any(tok in cmd for tok in [
        "r1088_merged","r1088_sim","local-r1088","vllm_chall_r1088",
        "wait_r1088","lean_chall_n80_r338_gpus67","chall_r1088",
        "p4221_r1088",
    ]):
        if "r1102" in cmd: continue
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
    if "r1102" in cmd or "r1100" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1088" not in cmd: print(f"SKIP TK {pid}"); continue
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
rm -rf /tmp/r1088_merged
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "TK warm"
python3 - <<'PY'
import json,time
from pathlib import Path
Path('/root/affine_data/r1088_refute_p4233.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1088","verdict":"REFUTE",
  "margin":0.0017745277061676451,"se":0.0012800390257433152,"z":1.3863075035053574,
  "n":80,"bar":0.0025600780514866304,"thought_median":168.0,"b_pass":0.45,
  "mult":0.6932,"king":"reign36","note":"p4233 → R1102 SoftCtx MidRank Hiβ Hyper HiLR isolate"
}, indent=2)+"\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1088_refute_reaped_p4233.done
nohup bash /root/mining_src/$EXP/lean_train_r338_gpus67_p4233.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4233_r1102_lean_outer.pid
sleep 25
echo "r1102 train.pid=$(cat /root/logs/r1102_train.pid 2>/dev/null || echo MISSING)"
tail -40 /root/logs/r1102_lean_warm.log 2>/dev/null || true
ps -eo pid,etime,cmd | grep -E 'r1102|train_dpo.*r1102' | grep -v grep | head -10
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4233_r1088_refute_r1102_armed.done
echo ARMED
REMOTE
