#!/usr/bin/env bash
# p4251 host-side: R1111 REFUTE → reap r338 :8003 GPUs4,5 → R1122 MidCtx HiRank Loβ Hyper HiLR TRAIN
# Do not touch R1102 idle chall :8002 on GPUs6,7 / teacher / king. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1122-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4251] sync $EXP → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1122 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4251] exact-PID reap R1111 chall :8003 pid=188410 (GPUs 4,5)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=188410
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1111_merged" && echo "$cmd" | grep -q -- "--port 8003"; then
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
    echo "reaped R1111 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1111 :8003 — abort"; exit 2
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
# Confirm R1102 chall still on 6,7
nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits || true
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1122-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1122-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r338_gpus45_p4251.sh >/root/logs/p4251_r1122_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4251_r1122_lean_outer.pid
sleep 3
echo OUTER_PID=$(cat /root/logs/p4251_r1122_lean_outer.pid)
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1122_train.pid ]]; then
    tp=$(cat /root/logs/r1122_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_PID=$tp
      head -40 /root/logs/r1122_lean_warm.log || true
      cat /root/affine_data/r1122_train_launched.json || true
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4251_r1111_refute_r1122_armed.done
      exit 0
    fi
  fi
  sleep 2
done
echo FATAL train not started
tail -80 /root/logs/r1122_lean_warm.log /root/logs/p4251_r1122_lean_outer.nohup 2>/dev/null || true
exit 1
REMOTE
