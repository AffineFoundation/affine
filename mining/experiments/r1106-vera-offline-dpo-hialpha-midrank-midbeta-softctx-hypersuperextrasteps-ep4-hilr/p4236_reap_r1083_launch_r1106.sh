#!/usr/bin/env bash
# p4236 host-side: R1083 REFUTE → reap r337 :8002 GPUs6,7 → R1106 Hyper HiLR TRAIN
# Do not touch R1094 n80 on GPUs 4,5 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1106-vera-offline-dpo-hialpha-midrank-midbeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=150.136.46.118
PORT=20300
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4236] sync $EXP → mine-r337"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1106 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4236] exact-PID reap R1083 chall :8002 pid=138127 (GPUs 6,7)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=138127
# Confirm it is still the R1083 chall on :8002
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1083_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
    # also kill children
    pids="$CHALL_PID $(pgrep -P $CHALL_PID 2>/dev/null || true)"
    for p in $pids; do kill "$p" 2>/dev/null || true; done
    for i in $(seq 1 40); do
      alive=0
      for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
      [[ $alive -eq 0 ]] && break
      sleep 1
    done
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    echo "reaped R1083 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1083 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi
# Wait VRAM 6,7 free
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 still busy; exit 1; }
# Confirm R1094 still alive on :8003
ss -lptn "sport = :8003" | head -3 || true
curl -sf -m 3 http://127.0.0.1:8003/v1/models >/dev/null && echo R1094_CHALL_OK || echo WARN_R1094_missing
chmod +x /root/mining_src/r1106-vera-offline-dpo-hialpha-midrank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1106-vera-offline-dpo-hialpha-midrank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r337_gpus67_p4236.sh >/root/logs/p4236_r1106_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4236_r1106_lean_outer.pid
sleep 3
echo OUTER_PID=$(cat /root/logs/p4236_r1106_lean_outer.pid)
# Wait for train pid
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1106_train.pid ]]; then
    tp=$(cat /root/logs/r1106_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_PID=$tp
      head -40 /root/logs/r1106_lean_warm.log || true
      cat /root/affine_data/r1106_train_launched.json || true
      exit 0
    fi
  fi
  sleep 2
done
echo FATAL train not started
tail -80 /root/logs/r1106_lean_warm.log /root/logs/p4236_r1106_lean_outer.nohup 2>/dev/null || true
exit 1
REMOTE
