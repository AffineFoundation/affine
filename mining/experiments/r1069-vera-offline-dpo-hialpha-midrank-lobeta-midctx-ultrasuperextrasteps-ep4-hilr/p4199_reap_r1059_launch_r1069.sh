#!/usr/bin/env bash
# p4199 host: exact-PID reap R1059 chall on crown :8003 GPUs4,5 → launch R1069 TRAIN. Never pkill -f.
# Leave R1066 TRAIN on GPUs1,3 and R1067 TRAIN on GPUs6,7 alone. Leave TK :8000/:8001 alone.
set -euo pipefail
EXP=r1069-vera-offline-dpo-hialpha-midrank-lobeta-midctx-ultrasuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining

tar -C "$ROOT" -czf /tmp/r1069_p4199.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_crown_gpus45_p4199.sh \
  wait_r1069_train_then_merge_p4199.sh \
  wait_r1069_merge_then_n80_p4199.sh \
  lean_chall_n80_crown_r1069_gpus45_p4199.sh

SSH_HOST=95.133.252.28
SSH_PORT=40298
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1069_p4199.tgz "root@${SSH_HOST}:/tmp/r1069_p4199.tgz"

ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1069-vera-offline-dpo-hialpha-midrank-lobeta-midctx-ultrasuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1069
tar -xzf /tmp/r1069_p4199.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1069/dpo_duel_reason.jsonl

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
  "/root/logs/vllm_chall_r1059.pid",
  "/root/logs/r1059_sim_wvk7.pid",
  "/root/logs/r1059_merge_then_n80.pid",
  "/root/logs/r1059_wait_merge.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass

try:
    out = subprocess.check_output(["ss","-lptn","sport = :8003"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
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
    if "/8001" in cmd or ":8001" in cmd: continue
    if any(tok in cmd for tok in [
        "r1059_merged","r1059_sim","local-r1059","vllm_chall_r1059",
        "wait_r1059","lean_chall_n80_crown_r1059","run_sim_duel.py"
    ]):
        if "run_sim_duel.py" in cmd and "r1059" not in cmd: continue
        # do not kill r1066/r1067 waiters
        if "r1066" in cmd or "r1067" in cmd: continue
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
    if "r1066" in cmd or "r1067" in cmd:
        print(f"SKIP sibling {pid}", flush=True); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1059" not in cmd: print(f"SKIP TK {pid}", flush=True); continue
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

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1059_refute_reaped_p4199.done

nohup bash /root/mining_src/$EXP/lean_train_crown_gpus45_p4199.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4199_r1069_lean_outer.pid
sleep 8
echo TRAIN_PID=$(cat /root/logs/r1069_train.pid 2>/dev/null || echo missing)
tail -n 30 /root/logs/r1069_lean_warm.log 2>/dev/null || true
pgrep -af 'train_dpo.py.*r1069|/root/r1069/train' | head -5 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5
REMOTE
