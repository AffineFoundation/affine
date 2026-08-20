#!/usr/bin/env bash
# p4203: exact-PID reap R1054 chall r924 :8003 GPUs1,3 → R1074 TRAIN. Never pkill -f.
# Leave R1073 TRAIN 6,7 / R1068 TRAIN 4,5 / TK alone.
set -euo pipefail
EXP=r1074-vera-offline-dpo-hialpha-lorank-hibeta-shortctx-ultrasuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
tar -C "$ROOT" -czf /tmp/r1074_p4203.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r924_gpus13_p4203.sh \
  wait_r1074_train_then_merge_p4203.sh \
  wait_r1074_merge_then_n80_p4203.sh \
  lean_chall_n80_r924_gpus13_p4203.sh
SSH_HOST=31.22.104.113; SSH_PORT=40300
SSH_OPTS="-i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1074_p4203.tgz "root@${SSH_HOST}:/tmp/r1074_p4203.tgz"
ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1074-vera-offline-dpo-hialpha-lorank-hibeta-shortctx-ultrasuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1074
tar -xzf /tmp/r1074_p4203.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1074/dpo_duel_reason.jsonl
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
  "/root/logs/vllm_chall_r1054.pid",
  "/root/logs/r1054_sim_wvk7.pid",
  "/root/logs/r1054_merge_then_n80.pid",
  "/root/logs/r1054_wait_merge.pid",
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
    if "r1073" in cmd or "r1068" in cmd: continue
    if any(tok in cmd for tok in [
        "r1054_merged","r1054_sim","local-r1054","vllm_chall_r1054",
        "wait_r1054","lean_chall_n80_r924_gpus13_p4183","run_sim_duel.py"
    ]):
        if "run_sim_duel.py" in cmd and "r1054" not in cmd: continue
        stop(pid, "argv")
out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]
    idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (1,3) if i in idx_to_uuid}
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
    if "r1073" in cmd or "r1068" in cmd: print(f"SKIP sibling {pid}", flush=True); continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1054" not in cmd: print(f"SKIP TK {pid}", flush=True); continue
    stop(pid, "gpu13")
print("reap done", flush=True)
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "wait free used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1054_refute_reaped_p4203.done
python3 - <<'PY'
import json,time
from pathlib import Path
p=Path("/root/affine_data/r1054_sim_result_reign36_wvk7.json")
d=json.loads(p.read_text()); v=d.get("verdict",{}); c=v.get("challenger",{})
m=float(v.get("margin") or 0); se=float(v.get("se") or 0)
summary={"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"axis":"R1054","pass":4203,
 "margin":m,"se":se,"z":v.get("z"),"n":v.get("n_paired_turns"),
 "thought_med":c.get("median_len_z"),"b_pass":c.get("b_gate_pass_rate"),
 "bar":max(2.0*se, float(v.get("min_margin") or 0.002)),
 "duel_params":v.get("duel_params"),"challenger_wins":v.get("challenger_wins")}
Path("/root/affine_data/r1054_decision_reign36_wvk7.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
PY
nohup bash /root/mining_src/$EXP/lean_train_r924_gpus13_p4203.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4203_r1074_lean_outer.pid
sleep 8
echo TRAIN_PID=$(cat /root/logs/r1074_train.pid 2>/dev/null || echo missing)
tail -25 /root/logs/r1074_lean_warm.log || true
ss -ltnp | grep -E ':800[0-9]' || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
REMOTE
echo "p4203 R1074 launch exit=$?"
