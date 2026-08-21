#!/usr/bin/env bash
# p4248 host-side: R1099 REFUTE → reap r938 :8002 GPUs2,3 → R1118 SoftCtx HiRank Midβ Hyper HiLR TRAIN
# Do not touch teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1118=r1118-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.255.28.21
PORT=20100
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4248] sync $E1118 → mine-r938"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1118 /root/r1118 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1118"/. "root@${HOST}:/root/mining_src/$E1118/"

echo "[p4248] exact-PID reap R1099 :8002 pid=52805"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=52805
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1099_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1099"
  else
    echo "FATAL pid $CHALL_PID is not R1099 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "VRAM2,3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 2,3 still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1118-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1118-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r938_gpus23_p4248.sh >/root/logs/p4248_r1118_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4248_r1118_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1118_train.pid ]]; then
    tp=$(cat /root/logs/r1118_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1118_PID=$tp
      head -40 /root/logs/r1118_lean_warm.log || true
      cat /root/affine_data/r1118_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4248_r1118_lean_outer.nohup /root/logs/r1118_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1118_TRAIN_OK
REMOTE
