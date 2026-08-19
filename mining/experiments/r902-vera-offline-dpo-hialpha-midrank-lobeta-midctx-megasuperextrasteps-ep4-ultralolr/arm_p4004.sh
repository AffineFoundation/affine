#!/usr/bin/env bash
set -euo pipefail
log(){ echo "[p4004-r902] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
if [[ -f /root/logs/p4004_r902_armed.done ]]; then echo ALREADY_ARMED; exit 0; fi
EXP=/root/mining_src/r902-vera-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
# Reap R885 chall exact pid
CHALL_PID=$(cat /root/logs/vllm_chall_r885.pid 2>/dev/null || true)
if [[ "$CHALL_PID" =~ ^[0-9]+$ ]] && kill -0 "$CHALL_PID" 2>/dev/null; then
  log "kill chall pid=$CHALL_PID"
  kill "$CHALL_PID" 2>/dev/null || true
  for _ in $(seq 1 40); do kill -0 "$CHALL_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$CHALL_PID" 2>/dev/null || true
fi
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2: idx_to_uuid[int(parts[0])]=parts[1]
uuids={idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in uuids: continue
    try: pid=int(parts[1])
    except: continue
    if pid<=1: continue
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except: cmd=""
    if "r885_merged" in cmd:
        print(f"kill r885 leftover pid={pid}", flush=True)
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
time.sleep(3)
print("reap done", flush=True)
PY
rm -f /root/logs/vllm_chall_r885.pid
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "VRAM4+5 used_mib=$used poll=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { log FATAL_BUSY; exit 1; }
mkdir -p /root/r902
test -s /root/r902/dpo_duel_reason.jsonl || cp -f "$EXP/dpo_duel_reason.jsonl" /root/r902/dpo_duel_reason.jsonl
bash "$EXP/lean_train_crown_gpus45_p4004.sh"
nohup bash "$EXP/wait_r902_train_then_merge_p4004.sh" >/root/logs/p4004_r902_wait.nohup 2>&1 &
echo $! >/root/logs/p4004_r902_wait.pid
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4004_r902_armed.done
echo TRAIN_PID=$(cat /root/logs/r902_train.pid)
echo WAIT_PID=$(cat /root/logs/p4004_r902_wait.pid)
tail -15 /root/logs/r902_lean_warm.log
sleep 2
tail -8 /root/logs/r902_train.nohup || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv
