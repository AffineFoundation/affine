#!/usr/bin/env bash
# p4230 host-side: R1082 REFUTE → reap r938 :8002 GPUs2,3 → R1099 SoftCtx TRAIN
set -euo pipefail
EXP=r1099-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-midlr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1099_p4230.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r938_gpus23_p4230.sh \
  wait_r1099_train_then_merge_p4230.sh \
  wait_r1099_merge_then_n80_p4230.sh \
  lean_chall_n80_r938_gpus23_p4230.sh
SSH_HOST=38.255.28.21; SSH_PORT=20100
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1099_p4230.tgz "root@${SSH_HOST}:/tmp/r1099_p4230.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1099-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1099 /root/affine_data
tar -xzf /tmp/r1099_p4230.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1099/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1082.pid","/root/logs/vllm_chall_r1082_p4216.pid",
    "/root/logs/r1082_sim_wvk7.pid","/root/logs/r1082_merge_then_n80.pid",
    "/root/logs/r1082_wait_merge.pid","/root/logs/p4216_r1082_lean_outer.pid",
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
        "r1082_merged","r1082_sim","local-r1082","vllm_chall_r1082",
        "wait_r1082","lean_chall_n80_r938_gpus23_p4216","chall_r1082",
        "p4216_r1082",
    ]):
        if "r1099" in cmd: continue
        stop(pid, "argv")
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]; idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (2,3) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: print(f"SKIP train {pid}"); continue
    if "r1099" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1082" not in cmd: print(f"SKIP TK {pid}"); continue
    stop(pid, "gpu23")
print("reap done", flush=True)
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# free old merge disk
rm -rf /tmp/r1082_merged
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "TK warm"
python3 - <<'PY'
import json,time
from pathlib import Path
Path('/root/affine_data/r1082_refute_p4230.json').write_text(json.dumps({
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo":"R1082","verdict":"REFUTE",
  "margin":0.0001724917173876434,"se":0.004629072538927657,"z":0.03726269483510875,
  "n":77,"bar":0.009258145077855315,"thought_median":187.0,"b_pass":0.4051724137931034,
  "mult":0.01863,"king":"reign36","note":"p4230 → R1099 SoftCtx isolate"
}, indent=2)+"\n")
PY
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1082_refute_reaped_p4230.done
nohup bash /root/mining_src/$EXP/lean_train_r938_gpus23_p4230.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4230_r1099_lean_outer.pid
sleep 20
echo "r1099 train.pid=$(cat /root/logs/r1099_train.pid 2>/dev/null || echo MISSING)"
tail -40 /root/logs/r1099_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4230_r1082_refute_r1099_armed.done
echo DONE
REMOTE
