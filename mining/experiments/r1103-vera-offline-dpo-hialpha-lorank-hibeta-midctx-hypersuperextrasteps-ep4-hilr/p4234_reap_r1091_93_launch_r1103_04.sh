#!/usr/bin/env bash
# p4234 host-side: R1091+R1093 REFUTE → reap crown :8004 GPUs1,3 + :8003 GPUs4,5 → R1103+R1104 Hyper HiLR TRAIN
# Do not touch R1101 train on GPUs 6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining

EXP3=r1103-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
EXP4=r1104-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-hilr

tar -C "$ROOT/$EXP3" -czf /tmp/r1103_p4234.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_crown_gpus13_p4234.sh \
  wait_r1103_train_then_merge_p4234.sh \
  wait_r1103_merge_then_n80_p4234.sh \
  lean_chall_n80_crown_r1103_gpus13_p4234.sh

tar -C "$ROOT/$EXP4" -czf /tmp/r1104_p4234.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_crown_gpus45_p4234.sh \
  wait_r1104_train_then_merge_p4234.sh \
  wait_r1104_merge_then_n80_p4234.sh \
  lean_chall_n80_crown_r1104_gpus45_p4234.sh

SSH_HOST=95.133.252.28; SSH_PORT=40298
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/.ralph/known_hosts -o ConnectTimeout=45 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1103_p4234.tgz /tmp/r1104_p4234.tgz "root@${SSH_HOST}:/tmp/"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP3=r1103-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
EXP4=r1104-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP3 /root/mining_src/$EXP4 /root/logs /root/r1103 /root/r1104 /root/affine_data
tar -xzf /tmp/r1103_p4234.tgz -C /root/mining_src/$EXP3
tar -xzf /tmp/r1104_p4234.tgz -C /root/mining_src/$EXP4
chmod +x /root/mining_src/$EXP3/*.sh /root/mining_src/$EXP4/*.sh
cp -f /root/mining_src/$EXP3/dpo_duel_reason.jsonl /root/r1103/dpo_duel_reason.jsonl
cp -f /root/mining_src/$EXP4/dpo_duel_reason.jsonl /root/r1104/dpo_duel_reason.jsonl

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
    "/root/logs/vllm_chall_r1091.pid","/root/logs/vllm_chall_r1091_p4223.pid",
    "/root/logs/r1091_sim_wvk7.pid","/root/logs/r1091_merge_then_n80.pid",
    "/root/logs/r1091_wait_merge.pid","/root/logs/p4223_r1091_lean_outer.pid",
    "/root/logs/vllm_chall_r1093.pid","/root/logs/vllm_chall_r1093_p4223.pid",
    "/root/logs/r1093_sim_wvk7.pid","/root/logs/r1093_merge_then_n80.pid",
    "/root/logs/r1093_wait_merge.pid","/root/logs/p4223_r1093_lean_outer.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass

for port in (8003, 8004):
    try:
        out = subprocess.check_output(["ss","-lptn",f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        out = ""
    for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
        stop(pid, f":{port}")

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
    # protect R1101 sibling on 6,7 / :8002
    if any(tok in cmd for tok in ["r1101","8002"]):
        if "r1091" not in cmd and "r1093" not in cmd: continue
    if any(tok in cmd for tok in [
        "r1091_merged","r1091_sim","local-r1091","vllm_chall_r1091",
        "wait_r1091","lean_chall_n80_crown_r1091","chall_r1091","p4223_r1091",
        "r1093_merged","r1093_sim","local-r1093","vllm_chall_r1093",
        "wait_r1093","lean_chall_n80_crown_r1093","chall_r1093","p4223_r1093",
    ]):
        if "r1103" in cmd or "r1104" in cmd or "r1101" in cmd: continue
        stop(pid, "argv")

out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]; idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (1,3,4,5) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: print(f"SKIP train {pid}"); continue
    if any(x in cmd for x in ["r1103","r1104","r1101"]): print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1091" not in cmd and "r1093" not in cmd: print(f"SKIP TK {pid}"); continue
    stop(pid, "gpu135")
print("reap done", flush=True)
PY

for pair in "1,3" "4,5"; do
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    echo "wait free $pair used_mib=$used iter=$i"
    [[ "$used" -lt 8192 ]] && break
    sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { echo FATAL still busy $pair; nvidia-smi; exit 1; }
done

rm -rf /tmp/r1091_merged /tmp/r1093_merged
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "TK warm"

python3 - <<'PY'
import json,time
from pathlib import Path
Path('/root/affine_data/r1091_refute_p4234.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1091","verdict":"REFUTE",
  "margin":0.0035519685203516116,"se":0.0038420607574471124,"z":0.9244956664120391,
  "n":76,"bar":0.007684121514894225,"thought_median":165.0,"b_pass":0.35526315789473684,
  "mult":0.4622,"king":"reign36","note":"p4234 → R1103 MidCtx LoRank Hiβ Hyper HiLR isolate"
}, indent=2)+"\n")
Path('/root/affine_data/r1093_refute_p4234.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1093","verdict":"REFUTE",
  "margin":0.0006458679185427504,"se":0.0031919712206540355,"z":0.20234139780571453,
  "n":80,"bar":0.006383942441308071,"thought_median":179.0,"b_pass":0.4375,
  "mult":0.1012,"king":"reign36","note":"p4234 → R1104 MidCtx MidRank Loβ Hyper HiLR isolate"
}, indent=2)+"\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1091_refute_reaped_p4234.done
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1093_refute_reaped_p4234.done

nohup bash /root/mining_src/$EXP3/lean_train_crown_gpus13_p4234.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4234_r1103_lean_outer.pid
nohup bash /root/mining_src/$EXP4/lean_train_crown_gpus45_p4234.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4234_r1104_lean_outer.pid
sleep 30
echo "r1103 train.pid=$(cat /root/logs/r1103_train.pid 2>/dev/null || echo MISSING)"
echo "r1104 train.pid=$(cat /root/logs/r1104_train.pid 2>/dev/null || echo MISSING)"
tail -20 /root/logs/r1103_lean_warm.log 2>/dev/null || true
tail -20 /root/logs/r1104_lean_warm.log 2>/dev/null || true
ps -eo pid,etime,cmd | grep -E 'r1103|r1104|train_dpo.*(r1103|r1104)' | grep -v grep | head -20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# confirm R1101 still alive
ps -eo pid,etime,cmd | grep r1101 | grep -v grep | head -5
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4234_r1091_93_refute_r1103_04_armed.done
echo ARMED
REMOTE
