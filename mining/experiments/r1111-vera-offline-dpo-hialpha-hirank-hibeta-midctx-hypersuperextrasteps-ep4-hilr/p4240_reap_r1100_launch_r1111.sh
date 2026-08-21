#!/usr/bin/env bash
# p4240 host-side: R1100 REFUTE idle → reap r338 :8003 GPUs4,5 → R1111 Hiβ Hyper HiLR TRAIN
# Do not touch R1102 train on GPUs 6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1111=r1111-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4240] sync $E1111 → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1111 /root/r1111 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1111"/. "root@${HOST}:/root/mining_src/$E1111/"

echo "[p4240] exact-PID reap R1100 :8003 pid=179309"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=179309
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1100_merged" && echo "$cmd" | grep -q -- "--port 8003"; then
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
    echo "reaped R1100"
  else
    echo "FATAL pid $CHALL_PID is not R1100 :8003 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8003"
  ss -lptn "sport = :8003" || true
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM4,5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 4,5 still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
if kill -0 177102 2>/dev/null; then echo R1102_TRAIN_OK pid=177102; else echo WARN R1102 train pid gone; fi

chmod +x /root/mining_src/r1111-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1111-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r338_gpus45_p4240.sh >/root/logs/p4240_r1111_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4240_r1111_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1111_train.pid ]]; then
    tp=$(cat /root/logs/r1111_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1111_PID=$tp
      head -40 /root/logs/r1111_lean_warm.log || true
      cat /root/affine_data/r1111_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4240_r1111_lean_outer.nohup /root/logs/r1111_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1111_TRAIN_OK
REMOTE
