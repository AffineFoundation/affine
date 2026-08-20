#!/usr/bin/env bash
# p4217: R1078 REFUTE m=-0.004523 ~-0.71× → reap r337 :8002 GPUs6,7 → R1083 SoftCtx Hyper MidLR TRAIN
set -euo pipefail
EXP=r1083-vera-offline-dpo-hialpha-midrank-midbeta-softctx-hypersuperextrasteps-ep4-midlr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1083_p4217.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r337_gpus67_p4217.sh \
  wait_r1083_train_then_merge_p4217.sh \
  wait_r1083_merge_then_n80_p4217.sh \
  lean_chall_n80_r337_gpus67_p4217.sh
SSH_HOST=150.136.46.118; SSH_PORT=20300
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/experiments/fleet-rent/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1083_p4217.tgz "root@${SSH_HOST}:/tmp/r1083_p4217.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1083-vera-offline-dpo-hialpha-midrank-midbeta-softctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1083
tar -xzf /tmp/r1083_p4217.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1083/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1078.pid","/root/logs/r1078_sim_wvk7.pid",
    "/root/logs/r1078_merge_then_n80.pid","/root/logs/r1078_wait_merge.pid",
    "/root/logs/p4207_r1078_lean_outer.pid","/root/logs/vllm_chall_r1078_p4207.pid",
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
        "r1078_merged","r1078_sim","local-r1078","vllm_chall_r1078",
        "wait_r1078","lean_chall_n80_r337_gpus67_p4207","chall_r1078",
        "p4207_r1078",
    ]):
        if "r1083" in cmd: continue
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
    if "r1083" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1078" not in cmd: print(f"SKIP TK {pid}"); continue
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
# free disk from stale R1078 merged weights before SoftCtx train
rm -rf /tmp/r1078_merged
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1078_refute_reaped_p4217.done
nohup bash /root/mining_src/$EXP/lean_train_r337_gpus67_p4217.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4217_r1083_lean_outer.pid
sleep 12
echo TRAIN_PID=$(cat /root/logs/r1083_train.pid 2>/dev/null || echo missing)
tail -n 40 /root/logs/r1083_lean_warm.log 2>/dev/null || true
pgrep -af 'train_dpo.py.*r1083' | head -3 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3,4,5,6,7
REMOTE
