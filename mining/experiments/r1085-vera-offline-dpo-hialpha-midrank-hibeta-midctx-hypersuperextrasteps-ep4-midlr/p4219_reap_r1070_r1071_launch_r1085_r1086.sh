#!/usr/bin/env bash
# p4219: R1070+R1071 REFUTE on idle mine-r339 → reap :8002/:8003 → R1085+R1086 TRAIN
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
E1085=r1085-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-midlr
E1086=r1086-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-midlr
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT/$E1085" -czf /tmp/r1085_p4219.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r339_gpus45_p4219.sh \
  wait_r1085_train_then_merge_p4219.sh \
  wait_r1085_merge_then_n80_p4219.sh \
  lean_chall_n80_r339_gpus45_p4219.sh
tar -C "$ROOT/$E1086" -czf /tmp/r1086_p4219.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r339_gpus67_p4219.sh \
  wait_r1086_train_then_merge_p4219.sh \
  wait_r1086_merge_then_n80_p4219.sh \
  lean_chall_n80_r339_gpus67_p4219.sh
SSH_HOST=23.153.44.20; SSH_PORT=40299
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/const/subnet120/mining/experiments/fleet-rent/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1085_p4219.tgz /tmp/r1086_p4219.tgz "root@${SSH_HOST}:/tmp/"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
E1085=r1085-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-midlr
E1086=r1086-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/mining_src/$E1085 /root/mining_src/$E1086 /root/logs /root/r1085 /root/r1086
tar -xzf /tmp/r1085_p4219.tgz -C /root/mining_src/$E1085
tar -xzf /tmp/r1086_p4219.tgz -C /root/mining_src/$E1086
chmod +x /root/mining_src/$E1085/*.sh /root/mining_src/$E1086/*.sh
cp -f /root/mining_src/$E1085/dpo_duel_reason.jsonl /root/r1085/dpo_duel_reason.jsonl
cp -f /root/mining_src/$E1086/dpo_duel_reason.jsonl /root/r1086/dpo_duel_reason.jsonl
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
    "/root/logs/vllm_chall_r1070.pid","/root/logs/r1070_sim_wvk7.pid",
    "/root/logs/r1070_merge_then_n80.pid","/root/logs/r1070_wait_merge.pid",
    "/root/logs/p4199_r1070_lean_outer.pid",
    "/root/logs/vllm_chall_r1071.pid","/root/logs/r1071_sim_wvk7.pid",
    "/root/logs/r1071_merge_then_n80.pid","/root/logs/r1071_wait_merge.pid",
    "/root/logs/p4201_r1071_lean_outer.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass
for port in (8002, 8003):
    try:
        out = subprocess.check_output(["ss","-lptn",f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
    except Exception: out=""
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
    if any(tok in cmd for tok in [
        "r1070_merged","r1070_sim","local-r1070","vllm_chall_r1070","wait_r1070","chall_r1070","p4199_r1070",
        "r1071_merged","r1071_sim","local-r1071","vllm_chall_r1071","wait_r1071","chall_r1071","p4201_r1071",
    ]):
        if "r1085" in cmd or "r1086" in cmd: continue
        stop(pid, "argv")
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]; idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (4,5,6,7) if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in want: continue
    pid=int(parts[1])
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: print(f"SKIP train {pid}"); continue
    if "r1085" in cmd or "r1086" in cmd: print(f"SKIP sibling {pid}"); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        print(f"SKIP TK {pid}"); continue
    stop(pid, "gpu457")
print("reap done", flush=True)
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
rm -rf /tmp/r1070_merged /tmp/r1071_merged
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1070_r1071_refute_reaped_p4219.done
nohup bash /root/mining_src/$E1085/lean_train_r339_gpus45_p4219.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4219_r1085_lean_outer.pid
nohup bash /root/mining_src/$E1086/lean_train_r339_gpus67_p4219.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4219_r1086_lean_outer.pid
sleep 15
echo R1085_TRAIN_PID=$(cat /root/logs/r1085_train.pid 2>/dev/null || echo missing)
echo R1086_TRAIN_PID=$(cat /root/logs/r1086_train.pid 2>/dev/null || echo missing)
tail -n 25 /root/logs/r1085_lean_warm.log 2>/dev/null || true
tail -n 25 /root/logs/r1086_lean_warm.log 2>/dev/null || true
pgrep -af 'train_dpo.py.*(r1085|r1086)' | head -6 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE
echo "p4219 R1085+R1086 launch exit=$?"
