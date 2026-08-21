#!/usr/bin/env bash
# p4270: R1126 REFUTE → reap :8002 → R1143 UltraLoLR TRAIN GPUs4,5; R1127 TP1 n80 GPU6.
# Never pkill -f. Do not touch teacher/king.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1143=r1143-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr
EXP1127=r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP1143"/*.sh "$ROOT/$EXP1127"/p4270_rearm_r1127_tp1_util085.sh
echo "[p4270] sync $EXP1143 + R1127 rearm → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP1143 /root/mining_src/$EXP1127 /root/r1143 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP1143"/. "root@${HOST}:/root/mining_src/$EXP1143/"
"${SCP[@]}" "$ROOT/$EXP1127/p4270_rearm_r1127_tp1_util085.sh" "root@${HOST}:/root/mining_src/$EXP1127/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2" port="$3"
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
# stop leftover R1126 sim if any
if [[ -f /root/logs/r1126_sim_wvk7.pid ]]; then
  sp=$(cat /root/logs/r1126_sim_wvk7.pid 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q r1126; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
  fi
fi
pid=$(cat /root/logs/vllm_chall_r1126.pid 2>/dev/null || echo 56449)
reap "$pid" r1126_merged 8002
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1126_merged; then reap "$p" r1126_merged 8002; fi
done < <(ss -lptn 'sport = :8002' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
# reap leftover compute apps on GPUs 4,5 (skip trains / TK)
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5}
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
        if "r1126_merged" not in cmd: continue
    kill.add(pid)
    print(f"gpu45 app pid={pid} cmd={cmd[:140]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu45 reaped", sorted(kill))
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM4+5=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy45; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1143-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/p4270_rearm_r1127_tp1_util085.sh
# launch R1143 train on 4,5
nohup bash /root/mining_src/r1143-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r339_gpus45_p4270.sh >/root/logs/p4270_r1143_outer.nohup 2>&1 &
echo $! >/root/logs/p4270_r1143_outer.pid
sleep 8
tp=$(cat /root/logs/r1143_train.pid 2>/dev/null || true)
echo "R1143_TRAIN_PID=$tp"
# re-arm R1127 TP1 n80 on GPU6 in background
nohup bash /root/mining_src/r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/p4270_rearm_r1127_tp1_util085.sh >/root/logs/p4270_r1127_outer.nohup 2>&1 &
echo $! >/root/logs/p4270_r1127_outer.pid
echo "R1127_REARM_OUTER=$(cat /root/logs/p4270_r1127_outer.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4270_r1126_refute_r1143_r1127_armed.done
echo ARMED
REMOTE
