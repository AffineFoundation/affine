#!/usr/bin/env bash
# p4198 host: exact-PID reap R1046 chall on r924 :8004 GPUs4,5 → launch R1068 TRAIN. Never pkill -f.
# Leave R1045 TRAIN on GPUs6,7 and R1054 TRAIN on GPUs1,3 alone. Leave TK :8000/:8001 alone.
set -euo pipefail
EXP=r1068-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining

tar -C "$ROOT" -czf /tmp/r1068_p4198.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r924_gpus45_p4198.sh \
  wait_r1068_train_then_merge_p4198.sh \
  wait_r1068_merge_then_n80_p4198.sh \
  lean_chall_n80_r924_gpus45_p4198.sh

SSH_HOST=31.22.104.113
SSH_PORT=40300
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1068_p4198.tgz "root@${SSH_HOST}:/tmp/r1068_p4198.tgz"

ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1068-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1068
tar -xzf /tmp/r1068_p4198.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1068/dpo_duel_reason.jsonl

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

for pf in [
  "/root/logs/vllm_chall_r1046.pid",
  "/root/logs/r1046_sim_wvk7.pid",
  "/root/logs/r1046_merge_then_n80.pid",
  "/root/logs/r1046_wait_merge.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass

try:
    out = subprocess.check_output(["ss","-lptn","sport = :8004"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
for pid in set(int(x) for x in re.findall(r"pid=(\d+)", out)):
    stop(pid, ":8004")

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
        "r1046_merged","r1046_sim","local-r1046","vllm_chall_r1046",
        "wait_r1046","lean_chall_n80_r924_gpus45_p4178","run_sim_duel.py"
    ]):
        if "run_sim_duel.py" in cmd and "r1046" not in cmd: continue
        stop(pid, "argv")

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
    if "r1045" in cmd or "r1054" in cmd:
        print(f"SKIP sibling {pid}", flush=True); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1046" not in cmd: print(f"SKIP TK {pid}", flush=True); continue
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

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1046_refute_reaped_p4198.done

nohup bash /root/mining_src/$EXP/lean_train_r924_gpus45_p4198.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4198_r1068_lean_outer.pid
sleep 8
echo "OUTER_PID=$(cat /root/logs/p4198_r1068_lean_outer.pid)"
echo "TRAIN_PID=$(cat /root/logs/r1068_train.pid 2>/dev/null || echo pending)"
tail -n 60 /root/logs/r1068_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,3,4,5,6,7
REMOTE
