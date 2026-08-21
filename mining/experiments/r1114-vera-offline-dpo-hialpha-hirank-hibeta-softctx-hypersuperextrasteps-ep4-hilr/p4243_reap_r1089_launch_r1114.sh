#!/usr/bin/env bash
# p4243 host-side: R1089 REFUTE → reap r924 :8002 GPUs6,7 → R1114 SoftCtx HiRank Hiβ Hyper HiLR TRAIN
# Do not touch R1113 TRAIN on 1,3 / R1112 TRAIN on 4,5 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1114=r1114-vera-offline-dpo-hialpha-hirank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4243] sync $E1114 → mine-r924"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1114 /root/r1114 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1114"/. "root@${HOST}:/root/mining_src/$E1114/"

echo "[p4243] exact-PID reap R1089 :8002 pid=134111"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=134111
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1089_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1089"
  else
    echo "FATAL pid $CHALL_PID is not R1089 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "VRAM6,7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
if [[ -f /root/logs/r1112_train.pid ]] && kill -0 "$(cat /root/logs/r1112_train.pid)" 2>/dev/null; then echo R1112_TRAIN_OK pid=$(cat /root/logs/r1112_train.pid); else echo WARN R1112 train gone; fi
if [[ -f /root/logs/r1113_train.pid ]] && kill -0 "$(cat /root/logs/r1113_train.pid)" 2>/dev/null; then echo R1113_TRAIN_OK pid=$(cat /root/logs/r1113_train.pid); else echo WARN R1113 train gone; fi

chmod +x /root/mining_src/r1114-vera-offline-dpo-hialpha-hirank-hibeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1114-vera-offline-dpo-hialpha-hirank-hibeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r924_gpus67_p4243.sh >/root/logs/p4243_r1114_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4243_r1114_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1114_train.pid ]]; then
    tp=$(cat /root/logs/r1114_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1114_PID=$tp
      head -40 /root/logs/r1114_lean_warm.log || true
      cat /root/affine_data/r1114_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4243_r1114_lean_outer.nohup /root/logs/r1114_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1114_TRAIN_OK
REMOTE
