#!/usr/bin/env bash
# p4252 host-side: R1105 REFUTE → reap r252 :8003 GPUs6,7 → R1123 ShortCtx HiRank MidLoβ Hyper HiLR TRAIN
# Do not touch R1110 train on GPUs 4,5 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1123-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.127.229.127
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4252] sync $EXP → mine-r252"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1123 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4252] exact-PID reap R1105 chall :8003 pid=182843 (GPUs 6,7)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=182843
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1105_merged" && echo "$cmd" | grep -q -- "--port 8003"; then
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
    echo "reaped R1105 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1105 :8003 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8003"
  ss -lptn "sport = :8003" || true
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "VRAM6+7=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 still busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# keep R1110 train alive
if [[ -f /root/logs/r1110_train.pid ]]; then
  tp=$(cat /root/logs/r1110_train.pid)
  kill -0 "$tp" 2>/dev/null && echo "R1110 TRAIN still alive pid=$tp" || echo "WARN R1110 train gone"
fi
chmod +x /root/mining_src/r1123-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1123-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r252_gpus67_p4252.sh >/root/logs/p4252_r1123_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4252_r1123_lean_outer.pid
sleep 3
echo "TRAIN_PID=$(cat /root/logs/r1123_train.pid 2>/dev/null || echo missing)"
tail -n 20 /root/logs/r1123_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4252_r1105_refute_r1123_armed.done
REMOTE
echo "[p4252] R1123 armed"
