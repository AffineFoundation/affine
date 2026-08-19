#!/usr/bin/env bash
# p4009: R894+R895 REFUTE → exact-PID reap challs → R908 MidLoβ ShortCtx + R909 Midβ ShortCtx TRAIN on R337 4–7
# Never pkill -f. Do not touch teacher 0,1 or king 2,3.
set -euo pipefail
log(){ echo "[p4009-r908r909] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
if [[ -f /root/logs/p4009_r908_r909_armed.done ]]; then echo ALREADY_ARMED; exit 0; fi
EXP908=/root/mining_src/r908-marsplan-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr
EXP909=/root/mining_src/r909-marsplan-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr

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

for pf in \
  /root/logs/vllm_chall_r894.pid /root/logs/vllm_chall_r895.pid \
  /root/logs/p4007_r894_n80.outer.pid /root/logs/p4007_r895_n80.outer.pid \
  /root/logs/r894_sim_wvk7.pid /root/logs/r895_sim_wvk7.pid \
  /root/logs/p4007_r895_wait_n80.outer.pid; do
  [[ -f "$pf" ]] || continue
  STOP "$(cat "$pf" 2>/dev/null || true)" "pidfile $pf"
  rm -f "$pf"
done

# Known live challs from p4007 if pidfiles already cleared
for pid in 118623 120786 118404 120629 123507 124675; do
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    case "$cmd" in
      *r894_merged*|*r895_merged*|*lean_chall_n80_r337*|*run_sim_duel.py*r894*|*run_sim_duel.py*r895*|*r894*|*r895*)
        STOP "$pid" "hardcoded leftover r894/r895"
        ;;
    esac
  fi
done

# Reap leftover merged-model vLLM on GPUs 4–7 only
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
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
    if "r894_merged" in cmd or "r895_merged" in cmd:
        print(f"kill leftover pid={pid}", flush=True)
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
time.sleep(3)
print("reap done", flush=True)
PY

for i in $(seq 1 90); do
  used=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "VRAM4-7 used_mib=$used poll=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { log FATAL_BUSY used=$used; exit 1; }

# Free disk from finished merges (keep adapters/data)
rm -rf /tmp/r894_merged /tmp/r895_merged || true

mkdir -p /root/r908 /root/r909
test -s /root/r908/dpo_duel_reason.jsonl || cp -f /root/r894/dpo_duel_reason.jsonl /root/r908/dpo_duel_reason.jsonl
test -s /root/r909/dpo_duel_reason.jsonl || cp -f /root/r895/dpo_duel_reason.jsonl /root/r909/dpo_duel_reason.jsonl || cp -f /root/r894/dpo_duel_reason.jsonl /root/r909/dpo_duel_reason.jsonl
chmod +x "$EXP908"/*.sh "$EXP909"/*.sh

bash "$EXP908/lean_train_r337_gpus67_p4009.sh"
nohup bash "$EXP908/wait_r908_train_then_merge_p4009.sh" >/root/logs/p4009_r908_wait.nohup 2>&1 &
echo $! >/root/logs/p4009_r908_wait.pid

bash "$EXP909/lean_train_r337_gpus45_p4009.sh"
nohup bash "$EXP909/wait_r909_train_then_merge_p4009.sh" >/root/logs/p4009_r909_wait.nohup 2>&1 &
echo $! >/root/logs/p4009_r909_wait.pid

date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4009_r908_r909_armed.done
echo TRAIN908=$(cat /root/logs/r908_train.pid)
echo WAIT908=$(cat /root/logs/p4009_r908_wait.pid)
echo TRAIN909=$(cat /root/logs/r909_train.pid)
echo WAIT909=$(cat /root/logs/p4009_r909_wait.pid)
tail -8 /root/logs/r908_lean_warm.log || true
tail -8 /root/logs/r909_lean_warm.log || true
sleep 4
tail -3 /root/logs/r908_train.nohup || true
tail -3 /root/logs/r909_train.nohup || true
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
