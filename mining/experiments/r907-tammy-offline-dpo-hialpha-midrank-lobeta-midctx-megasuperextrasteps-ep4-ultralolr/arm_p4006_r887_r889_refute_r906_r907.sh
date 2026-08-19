#!/usr/bin/env bash
# p4006: R887+R889 REFUTE → reap challs → R906 TRAIN GPUs2,3 + R907 TRAIN GPUs4,5
# Never pkill -f. Leave teacher0 king1 and R905 on 6,7 alone.
set -euo pipefail
log(){ echo "[p4006-r906r907] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
if [[ -f /root/logs/p4006_r906_r907_armed.done ]]; then echo ALREADY_ARMED; exit 0; fi
EXP906=/root/mining_src/r906-tammy-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr
EXP907=/root/mining_src/r907-tammy-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr

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

for pf in /root/logs/vllm_chall_r887.pid /root/logs/vllm_chall_r889.pid \
          /root/logs/p4004_r887_chall.outer.pid /root/logs/p4003_r889_chall.outer.pid \
          /root/logs/r887_sim_wvk7.pid /root/logs/r889_sim_wvk7.pid \
          /root/logs/r887_lean_outer.pid /root/logs/r889_lean_outer.pid; do
  [[ -f "$pf" ]] || continue
  STOP "$(cat "$pf" 2>/dev/null || true)" "pidfile $pf"
  rm -f "$pf"
done
for pid in 49775 49671 45831 45620 51231 48702; do
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    case "$cmd" in
      *r887_merged*|*r889_merged*|*lean_chall_n80_brave_gpus2*|*lean_chall_n80_brave_gpus4*)
        STOP "$pid" "hardcoded leftover"
        ;;
    esac
  fi
done

python3 - <<'PY'
import os, signal, subprocess, time
want={2,4}
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
    if "r887_merged" in cmd or "r889_merged" in cmd:
        print(f"kill leftover pid={pid}", flush=True)
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
time.sleep(3)
print("reap done", flush=True)
PY

for i in $(seq 1 90); do
  used23=$(nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "VRAM23=$used23 VRAM45=$used45 poll=$i"
  [[ "$used23" -lt 8192 && "$used45" -lt 8192 ]] && break
  sleep 2
done
used23=$(nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "$used23" -lt 8192 && "$used45" -lt 8192 ]] || { log FATAL_BUSY; exit 1; }

mkdir -p /root/r906 /root/r907
cp -f "$EXP906/dpo_duel_reason.jsonl" /root/r906/dpo_duel_reason.jsonl
cp -f "$EXP907/dpo_duel_reason.jsonl" /root/r907/dpo_duel_reason.jsonl
chmod +x "$EXP906"/*.sh "$EXP907"/*.sh

bash "$EXP906/lean_train_brave_gpus23_p4006.sh"
bash "$EXP907/lean_train_brave_gpus45_p4006.sh"
nohup bash "$EXP906/wait_r906_train_then_merge_p4006.sh" >/root/logs/p4006_r906_wait.nohup 2>&1 &
echo $! >/root/logs/p4006_r906_wait.pid
nohup bash "$EXP907/wait_r907_train_then_merge_p4006.sh" >/root/logs/p4006_r907_wait.nohup 2>&1 &
echo $! >/root/logs/p4006_r907_wait.pid
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4006_r906_r907_armed.done
echo TRAIN906=$(cat /root/logs/r906_train.pid) TRAIN907=$(cat /root/logs/r907_train.pid)
tail -6 /root/logs/r906_lean_warm.log; tail -6 /root/logs/r907_lean_warm.log
sleep 3
nvidia-smi -i 2,3,4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv
