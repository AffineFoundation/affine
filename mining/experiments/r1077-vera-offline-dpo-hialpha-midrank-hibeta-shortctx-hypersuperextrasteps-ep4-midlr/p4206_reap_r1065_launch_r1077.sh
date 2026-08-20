#!/usr/bin/env bash
# p4206: R1065 REFUTE causality_fail B=0.292 ~0.015× → R1077 HyperExtra MidLR on r338 GPUs 6,7
set -euo pipefail
EXP=r1077-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-midlr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1077_p4206.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r338_gpus67_p4206.sh \
  wait_r1077_train_then_merge_p4206.sh \
  wait_r1077_merge_then_n80_p4206.sh \
  lean_chall_n80_r338_gpus67_p4206.sh
SSH_HOST=95.133.253.90; SSH_PORT=40099
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1077_p4206.tgz "root@${SSH_HOST}:/tmp/r1077_p4206.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1077-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1077
tar -xzf /tmp/r1077_p4206.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1077/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1065.pid","/root/logs/r1065_sim_wvk7.pid",
    "/root/logs/r1065_merge_then_n80.pid","/root/logs/r1065_wait_merge.pid",
    "/root/logs/p4196_r1065_lean_outer.pid","/root/logs/vllm_chall_r1065_p4196.pid",
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
    if any(tok in cmd for tok in [
        "r1065_merged","r1065_sim","local-r1065","vllm_chall_r1065",
        "wait_r1065","lean_chall_n80_r338_gpus67_p4196","run_sim_duel.py","chall_r1065",
    ]):
        if "run_sim_duel.py" in cmd and "r1065" not in cmd and "r1077" not in cmd: continue
        if "r1072" in cmd or "r1077" in cmd: continue
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
    if "r1072" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1065" not in cmd: print(f"SKIP TK {pid}"); continue
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
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1065_refute_reaped_p4206.done
# write decision stamp
python3 - <<'PY'
import json,time
from pathlib import Path
Path("/root/affine_data/r1065_decision_p4206.json").write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis":"r1065",
  "verdict":"REFUTE",
  "rejection_reason":"causality_fail",
  "margin":6.588803828977946e-05,
  "se":0.0022703726299340777,
  "z":0.02902080364300929,
  "n":78,
  "bar":0.004540745259868155,
  "mult":0.014510401821504645,
  "thought_med":185.0,
  "b_pass":0.2923728813559322,
  "next":"r1077 HyperExtra MidLR 38400",
}, indent=2)+"\n")
PY
nohup bash /root/mining_src/$EXP/lean_train_r338_gpus67_p4206.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4206_r1077_lean_outer.pid
sleep 8
echo TRAIN_PID=$(cat /root/logs/r1077_train.pid 2>/dev/null || echo missing)
tail -n 25 /root/logs/r1077_lean_warm.log 2>/dev/null || true
pgrep -af 'train_dpo.py.*r1077' | head -3 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 6,7
REMOTE
