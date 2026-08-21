#!/usr/bin/env bash
# p4274: R1123 REFUTE → exact-PID reap :8003 → R1153 UltraLoLR TRAIN GPUs6,7.
# Never pkill -f. Do not touch teacher/king or R1138 on GPUs4,5.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1153=r1153-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.127.229.127
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP1153"/*.sh
echo "[p4274] sync $EXP1153 → mine-r252"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP1153 /root/r1153 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP1153"/. "root@${HOST}:/root/mining_src/$EXP1153/"
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
if [[ -f /root/logs/r1123_sim_wvk7.pid ]]; then
  sp=$(cat /root/logs/r1123_sim_wvk7.pid 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q r1123; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
  fi
fi
pid=$(cat /root/logs/vllm_chall_r1123.pid 2>/dev/null || echo 194702)
reap "$pid" r1123_merged
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1123_merged; then reap "$p" r1123_merged; fi
done < <(ss -lptn 'sport = :8003' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
python3 - <<'PY'
import os, signal, subprocess, time
want={6,7}
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
        if "r1123_merged" not in cmd and "r1153_merged" not in cmd: continue
    if "r1138" in cmd: continue
    kill.add(pid)
    print(f"gpu67 app pid={pid} cmd={cmd[:140]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu67 apps cleared")
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4274] wait free gpus6,7 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
if ! kill -0 "$(cat /root/logs/r1138_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1138 train not alive; fi
chmod +x /root/mining_src/r1153-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1153-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r252_gpus67_p4274.sh >/root/logs/p4274_r1153_outer.nohup 2>&1 &
echo $! >/root/logs/p4274_r1153_outer.pid
sleep 8
echo "=== verify ==="
cat /root/logs/r1153_train.pid 2>/dev/null || true
head -5 /root/logs/r1153_lean_warm.log 2>/dev/null || true
ps -p "$(cat /root/logs/r1153_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null || { echo FATAL train not up; tail -40 /root/logs/r1153_lean_warm.log; exit 1; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4274_r1123_refute_r1153_armed.done
echo "[p4274] R1153 TRAIN armed"
REMOTE
