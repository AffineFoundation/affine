#!/usr/bin/env bash
# p4253 host-side: R1106 REFUTE → reap r337 :8002 GPUs6,7 → R1125 SoftCtx LoRank Midβ Hyper HiLR TRAIN
# Do not touch R1119 train on GPUs 4,5 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1125-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=150.136.46.118
PORT=20300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4253] sync $EXP → mine-r337"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1125 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4253] exact-PID reap R1106 chall :8002 pid=150962 (GPUs 6,7)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=150962
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1106_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1106 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1106 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
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
# keep R1119 train alive
if [[ -f /root/logs/r1119_train.pid ]]; then
  tp=$(cat /root/logs/r1119_train.pid)
  kill -0 "$tp" 2>/dev/null && echo "R1119 TRAIN still alive pid=$tp" || echo "WARN R1119 train gone"
fi
chmod +x /root/mining_src/r1125-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1125-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r337_gpus67_p4253.sh >/root/logs/p4253_r1125_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4253_r1125_lean_outer.pid
sleep 5
echo "TRAIN_PID=$(cat /root/logs/r1125_train.pid 2>/dev/null || echo missing)"
tail -n 40 /root/logs/r1125_lean_warm.log 2>/dev/null || true
cat /root/affine_data/r1125_train_launched.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4253_r1106_refute_r1125_armed.done
REMOTE
echo "[p4253] R1125 armed"
