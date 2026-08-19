#!/usr/bin/env bash
# p4006: R890 REFUTE → exact-PID reap chall → R905 TRAIN+wait→merge on brave GPUs 6,7
# Never pkill -f. Do not touch teacher 0, king 1, R887 GPU2, R889 GPU4.
set -euo pipefail
log(){ echo "[p4006-r905] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
if [[ -f /root/logs/p4006_r905_armed.done ]]; then echo ALREADY_ARMED; exit 0; fi
EXP=/root/mining_src/r905-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr

STOP() {
  local pid=$1 why=$2
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "stop pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# Reap R890 chall / outer / sim exact PIDs only
for pf in /root/logs/vllm_chall_r890.pid /root/logs/p4003_r890_chall.outer.pid \
          /root/logs/r890_sim_wvk7.pid /root/logs/r890_lean_outer.pid; do
  [[ -f "$pf" ]] || continue
  STOP "$(cat "$pf" 2>/dev/null || true)" "pidfile $pf"
  rm -f "$pf"
done
# Known live chall from p4003 if pidfile already cleared
for pid in 45836 45621; do
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    case "$cmd" in
      *r890_merged*|*lean_chall_n80_brave_gpus6*|*r890*)
        STOP "$pid" "hardcoded leftover r890"
        ;;
    esac
  fi
done

# Also reap any leftover r890_merged vLLM by UUID on GPU6 only
python3 - <<'PY'
import os, signal, subprocess, time
want={6}
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
    if "r890_merged" in cmd:
        print(f"kill leftover pid={pid}", flush=True)
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
time.sleep(3)
print("reap done", flush=True)
PY

for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "VRAM6-7 used_mib=$used poll=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { log FATAL_BUSY used=$used; exit 1; }

mkdir -p /root/r905
test -s /root/r905/dpo_duel_reason.jsonl || cp -f /root/r890/dpo_duel_reason.jsonl /root/r905/dpo_duel_reason.jsonl
chmod +x "$EXP"/*.sh

bash "$EXP/lean_train_brave_gpus67_p4006.sh"
nohup bash "$EXP/wait_r905_train_then_merge_p4006.sh" >/root/logs/p4006_r905_wait.nohup 2>&1 &
echo $! >/root/logs/p4006_r905_wait.pid
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4006_r905_armed.done
echo TRAIN905=$(cat /root/logs/r905_train.pid)
echo WAIT905=$(cat /root/logs/p4006_r905_wait.pid)
tail -12 /root/logs/r905_lean_warm.log
sleep 3
tail -5 /root/logs/r905_train.nohup || true
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
