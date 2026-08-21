#!/usr/bin/env bash
# p4239 host-side: R1090 REFUTE → reap r252 :8002 GPUs4,5 → R1110 Hyper HiLR TRAIN
# Do not touch R1105 train on GPUs 6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1110=r1110-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.127.229.127
PORT=40299
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4239] sync $E1110 → mine-r252"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1110 /root/r1110 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1110"/. "root@${HOST}:/root/mining_src/$E1110/"

echo "[p4239] exact-PID reap R1090 :8002 pid=176314"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=176314
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1090_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
    # collect descendants
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
    echo "reaped R1090"
  else
    echo "FATAL pid $CHALL_PID is not R1090 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM4,5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 4,5 still busy; exit 1; }

# Confirm TK + R1105 still up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
if kill -0 174547 2>/dev/null; then echo R1105_TRAIN_OK pid=174547; else echo WARN R1105 train pid gone; fi

chmod +x /root/mining_src/r1110-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1110-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r252_gpus45_p4239.sh >/root/logs/p4239_r1110_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4239_r1110_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1110_train.pid ]]; then
    tp=$(cat /root/logs/r1110_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1110_PID=$tp
      head -40 /root/logs/r1110_lean_warm.log || true
      cat /root/affine_data/r1110_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4239_r1110_lean_outer.nohup /root/logs/r1110_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1110_TRAIN_OK
REMOTE
