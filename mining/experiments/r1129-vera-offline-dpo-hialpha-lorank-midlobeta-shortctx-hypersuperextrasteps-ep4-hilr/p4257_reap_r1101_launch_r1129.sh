#!/usr/bin/env bash
# p4257 host-side: R1101 REFUTE → reap crown :8002 GPUs6,7 → R1129 ShortCtx LoRank MidLoβ Hyper HiLR TRAIN
# Do not touch R1116/R1117 trains / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1129-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

chmod +x "$ROOT/$EXP"/*.sh

echo "[p4257] sync $EXP → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1129 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

echo "[p4257] exact-PID reap R1101 chall :8002 pid=279494 (GPUs 6,7)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=279494
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1101_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1101 chall"
  else
    echo "FATAL pid $CHALL_PID is not R1101 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi
# also stop leftover sim if any (exact match)
SIM=$(ps -eo pid=,args= | awk '/run_sim_duel.py .*local-r1101-reign36-wvk7/ && !/awk/ {print $1}')
for p in $SIM; do
  echo "stop leftover sim pid=$p"
  kill "$p" 2>/dev/null || true
done
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
# keep R1116/R1117 train alive
ps -eo pid,args | grep -E 'r1116/train|r1117/train' | grep -v grep | head -5 || echo "WARN sibling trains not seen"
chmod +x /root/mining_src/r1129-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1129-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_crown_gpus67_p4257.sh >/root/logs/p4257_r1129_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4257_r1129_lean_outer.pid
sleep 5
echo "TRAIN_PID=$(cat /root/logs/r1129_train.pid 2>/dev/null || echo missing)"
tail -n 30 /root/logs/r1129_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4257_r1101_refute_r1129_armed.done
REMOTE
echo "[p4257] R1129 armed"
