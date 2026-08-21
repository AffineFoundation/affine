#!/usr/bin/env bash
# p4250 host-side: R1107 REFUTE → reap r337 :8003 GPUs4,5 → R1119 ShortCtx MidRank Midβ Hyper HiLR TRAIN
# Do not touch R1106 TRAIN on GPUs 6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1119-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=150.136.46.118
PORT=20300
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4250] sync $EXP → mine-r337"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1119 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4250] exact-PID reap R1107 chall :8003 pid=147283 (GPUs 4,5)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=147283
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1107_merged" && echo "$cmd" | grep -q -- "--port 8003"; then
    pids="$CHALL_PID"
    kids=$(pgrep -P $CHALL_PID 2>/dev/null || true)
    for k in $kids; do
      pids="$pids $k"
      gkids=$(pgrep -P $k 2>/dev/null || true)
      for g in $gkids; do pids="$pids $g"; done
    done
    echo "kill set: $pids"
    for p in $pids; do kill "$p" 2>/dev/null || true; done
    for i in $(seq 1 40); do
      alive=0
      for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
      [[ $alive -eq 0 ]] && break
      sleep 1
    done
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    echo "reaped R1107 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1107 :8003 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8003"
  ss -lptn "sport = :8003" || true
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 4,5 still busy; exit 1; }
# Confirm R1106 TRAIN still on GPUs 6,7
nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits || true
if [[ -f /root/logs/r1106_train.pid ]]; then
  tp=$(cat /root/logs/r1106_train.pid)
  kill -0 "$tp" 2>/dev/null && echo R1106_TRAIN_OK pid=$tp || echo WARN_R1106_missing
fi
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1119-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1119-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r337_gpus45_p4250.sh >/root/logs/p4250_r1119_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4250_r1119_lean_outer.pid
sleep 3
echo OUTER_PID=$(cat /root/logs/p4250_r1119_lean_outer.pid)
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1119_train.pid ]]; then
    tp=$(cat /root/logs/r1119_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_PID=$tp
      head -40 /root/logs/r1119_lean_warm.log || true
      cat /root/affine_data/r1119_train_launched.json || true
      exit 0
    fi
  fi
  sleep 2
done
echo FATAL train not started
tail -80 /root/logs/r1119_lean_warm.log /root/logs/p4250_r1119_lean_outer.nohup 2>/dev/null || true
exit 1
REMOTE
