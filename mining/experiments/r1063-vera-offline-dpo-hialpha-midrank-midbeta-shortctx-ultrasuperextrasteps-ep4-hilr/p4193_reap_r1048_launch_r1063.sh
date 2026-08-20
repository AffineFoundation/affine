#!/usr/bin/env bash
# p4193 host: exact-PID reap R1048 chall on r337 :8002 GPUs6,7 → launch R1063 TRAIN. Never pkill -f.
# Leave R1047 on :8003 GPUs4,5 alone.
set -euo pipefail
EXP=r1063-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-hilr
ROOT=/home/const/subnet120/mining/experiments/$EXP
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining

tar -C "$ROOT" -czf /tmp/r1063_p4193.tgz \
  train_dpo.py merge_lora.py dpo_duel_reason.jsonl \
  lean_train_r337_gpus67_p4193.sh \
  wait_r1063_train_then_merge_p4193.sh \
  wait_r1063_merge_then_n80_p4193.sh \
  lean_chall_n80_r337_gpus67_p4193.sh

SSH_HOST=150.136.46.118
SSH_PORT=20300
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes"
scp $SSH_OPTS -P "$SSH_PORT" /tmp/r1063_p4193.tgz "root@${SSH_HOST}:/tmp/r1063_p4193.tgz"

ssh $SSH_OPTS -p "$SSH_PORT" "root@${SSH_HOST}" bash -s <<'REMOTE'
set -euo pipefail
EXP=r1063-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-hilr
mkdir -p /root/mining_src/$EXP /root/logs /root/r1063
tar -xzf /tmp/r1063_p4193.tgz -C /root/mining_src/$EXP
chmod +x /root/mining_src/$EXP/*.sh
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1063/dpo_duel_reason.jsonl

# Exact-PID reap: R1048 sim + chall vllm + waiters (NOT TK, NOT R1047)
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
  "/root/logs/vllm_chall_r1048.pid",
  "/root/logs/r1048_sim_wvk7.pid",
  "/root/logs/r1048_merge_then_n80.pid",
  "/root/logs/r1048_wait_merge.pid",
  "/root/logs/p4192_r1048_lean_n80.pid",
]:
    if os.path.exists(pf):
        try: stop(int(open(pf).read().strip()), pf)
        except Exception as e: print("pidf", pf, e)
        try: os.remove(pf)
        except Exception: pass

try:
    out = subprocess.check_output(["ss","-lptn","sport = :8002"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ""
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
    if "GLM-4.5-Air" in cmd and ("8000" in cmd or ":8000" in cmd): continue
    if "/8001" in cmd or ":8001" in cmd: continue
    if "r1047" in cmd or "8003" in cmd: continue
    if any(tok in cmd for tok in [
        "r1048_merged","r1048_sim","local-r1048","vllm_chall_r1048",
        "wait_r1048","lean_chall_n80_r337_gpus67_p4179","lean_chall_n80_r337_gpus67"
    ]):
        if "lean_chall" in cmd and "r1047" in cmd: continue
        if "run_sim_duel.py" in cmd and "r1048" not in cmd: continue
        stop(pid, "argv")
    if "run_sim_duel.py" in cmd and "r1048" in cmd:
        stop(pid, "sim-r1048")

out = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"], text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    a,b=[p.strip() for p in line.split(",")]
    idx_to_uuid[int(a)]=b
want={idx_to_uuid[i] for i in (6,7) if i in idx_to_uuid}
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
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1048" not in cmd: print(f"SKIP TK {pid}", flush=True); continue
    if "r1047" in cmd:
        print(f"SKIP r1047 {pid}", flush=True); continue
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

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1048_refute_reaped_p4193.done
python3 - <<'PY'
import json,time
from pathlib import Path
src=Path("/root/affine_data/r1048_sim_result_reign36_wvk7.json")
d=json.loads(src.read_text()) if src.exists() else {}
v=d.get("verdict",{})
chal=v.get("challenger",{})
se=float(v.get("se") or 0)
note={
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis":"R1048","status":"REFUTE",
  "margin": v.get("margin"), "se": v.get("se"), "z": v.get("z"),
  "n_paired": v.get("n_paired_turns"),
  "bar": max(2*se, float(v.get("min_margin") or 0.002)),
  "thought_med": chal.get("median_len_z"),
  "b_pass": chal.get("b_gate_pass_rate"),
  "k": v.get("duel_params",{}).get("n_teacher_samples"),
  "tau": v.get("duel_params",{}).get("tau"),
  "wins": v.get("challenger_wins"),
  "next":"R1063 ShortCtx MidRank Midβ Ultra HiLR"
}
Path("/root/affine_data/r1048_refute_p4193.json").write_text(json.dumps(note,indent=2)+"\n")
print(json.dumps(note,indent=2))
PY

nohup bash /root/mining_src/$EXP/lean_train_r337_gpus67_p4193.sh >/root/logs/p4193_r1063_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4193_r1063_lean_outer.pid
sleep 4
echo OUTER_PID=$(cat /root/logs/p4193_r1063_lean_outer.pid)
echo TRAIN_PID=$(cat /root/logs/r1063_train.pid 2>/dev/null || echo pending)
tail -30 /root/logs/r1063_lean_warm.log 2>/dev/null || tail -30 /root/logs/p4193_r1063_lean_outer.nohup
nvidia-smi --query-gpu=index,memory.used --format=csv
# Confirm R1047 still alive
ss -tlnp | grep -E '800[0-3]' || true
ps -eo pid,etime,cmd | grep -E 'r1047_sim|r1047_merged|run_sim_duel.*r1047' | grep -v grep | head -5 || true
REMOTE

echo "HOST launch script finished"
