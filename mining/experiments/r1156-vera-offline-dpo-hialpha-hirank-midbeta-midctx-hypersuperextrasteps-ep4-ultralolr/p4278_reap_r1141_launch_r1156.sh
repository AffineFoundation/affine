#!/usr/bin/env bash
# p4278: R1141 REFUTE → exact-PID reap :8003 → R1156 Midβ UltraLoLR TRAIN GPUs3,4.
# Never pkill -f. Do not touch teacher/king or R1142/R1144 trains.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1156=r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP1156"/*.sh
echo "[p4278] sync $EXP1156 → mine-r340"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP1156 /root/r1156 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP1156"/. "root@${HOST}:/root/mining_src/$EXP1156/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
# stop lean wrapper if still holding
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'lean_chall_n80_r340_gpus34_p4265'; then
    reap "$p" lean_chall_n80_r340_gpus34_p4265
  fi
  if echo "$cmd" | grep -q 'wait_r1141_merge_then_n80'; then
    reap "$p" wait_r1141_merge_then_n80
  fi
done < <(ps -eo pid=,args= | awk '/lean_chall_n80_r340_gpus34_p4265|wait_r1141_merge_then_n80/{print $1}')
if [[ -f /root/logs/r1141_sim_wvk7.pid ]]; then
  sp=$(cat /root/logs/r1141_sim_wvk7.pid 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q r1141; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
  fi
fi
pid=$(cat /root/logs/vllm_chall_r1141.pid 2>/dev/null || echo 70069)
reap "$pid" r1141_merged
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1141_merged; then reap "$p" r1141_merged; fi
done < <(ss -lptn 'sport = :8003' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
python3 - <<'PY'
import os, signal, subprocess, time
want={3,4}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2: idx_to_uuid[int(parts[0])]=parts[1]
uuids={idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True)
kill=set()
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in uuids: continue
    try: pid=int(parts[1])
    except ValueError: continue
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1141_merged" not in cmd and "r1156_merged" not in cmd: continue
    if "r1142" in cmd or "r1144" in cmd: continue
    kill.add(pid)
    print(f"gpu34 app pid={pid} cmd={cmd[:140]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu34 apps cleared")
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4278] wait free gpus3,4 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# siblings should still be training
if ! kill -0 "$(cat /root/logs/r1142_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1142 train not alive; fi
if ! kill -0 "$(cat /root/logs/r1144_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1144 train not alive; fi
chmod +x /root/mining_src/r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r340_gpus34_p4278.sh >/root/logs/p4278_r1156_outer.nohup 2>&1 &
echo $! >/root/logs/p4278_r1156_outer.pid
sleep 10
echo "=== verify ==="
cat /root/logs/r1156_train.pid 2>/dev/null || true
head -8 /root/logs/r1156_lean_warm.log 2>/dev/null || true
ps -p "$(cat /root/logs/r1156_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null || { echo FATAL train not up; tail -40 /root/logs/r1156_lean_warm.log; exit 1; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4278_r1141_refute_r1156_armed.done
echo "[p4278] R1156 TRAIN armed"
REMOTE
