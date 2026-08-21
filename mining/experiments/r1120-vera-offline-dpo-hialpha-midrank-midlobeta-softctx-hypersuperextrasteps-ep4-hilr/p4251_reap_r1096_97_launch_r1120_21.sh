#!/usr/bin/env bash
# p4251 host-side: R1096+R1097 REFUTE → reap r340 :8002/:8003 → R1120+R1121 HiLR TRAIN
# Never pkill -f. Keep teacher:8000 GPU0 / king:8001 GPU5.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

E20=r1120-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr
E21=r1121-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr

echo "[p4251] sync $E20 + $E21 → mine-r340"
"${SSH[@]}" "mkdir -p /root/mining_src/$E20 /root/mining_src/$E21 /root/r1120 /root/r1121 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E20"/. "root@${HOST}:/root/mining_src/$E20/"
"${SCP[@]}" -r "$ROOT/$E21"/. "root@${HOST}:/root/mining_src/$E21/"

echo "[p4251] exact-PID reap R1096 :8002 pid=43327 (GPU1) + R1097 :8003 pid=39045 (GPU3)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap_one() {
  local CHALL_PID=$1 WANT_MERGED=$2 WANT_PORT=$3
  if kill -0 "$CHALL_PID" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
    echo "reap pid=$CHALL_PID cmd=$cmd"
    if echo "$cmd" | grep -q "$WANT_MERGED" && echo "$cmd" | grep -q -- "--port $WANT_PORT"; then
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
      echo "reaped $WANT_MERGED :$WANT_PORT"
    else
      echo "FATAL pid $CHALL_PID is not $WANT_MERGED :$WANT_PORT — abort"; exit 2
    fi
  else
    echo "pid $CHALL_PID already gone — check :$WANT_PORT"
    ss -lptn "sport = :$WANT_PORT" || true
  fi
}
# also stop finished sims if lingering
for sp in 44907 45041; do
  if kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q run_sim_duel; then kill "$sp" 2>/dev/null || true; fi
  fi
done
reap_one 43327 r1096_merged 8002
reap_one 39045 r1097_merged 8003
for i in $(seq 1 90); do
  used12=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
  used34=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "VRAM1+2=$used12 VRAM3+4=$used34 iter=$i"
  [[ "$used12" -lt 8192 && "$used34" -lt 8192 ]] && break
  sleep 2
done
used12=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
used34=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used12" -lt 8192 ]] || { echo FATAL GPUs 1,2 still busy; exit 1; }
[[ "$used34" -lt 8192 ]] || { echo FATAL GPUs 3,4 still busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1120-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1121-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1120-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r340_gpus12_p4251.sh >/root/logs/p4251_r1120_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4251_r1120_lean_outer.pid
nohup bash /root/mining_src/r1121-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r340_gpus34_p4251.sh >/root/logs/p4251_r1121_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4251_r1121_lean_outer.pid
sleep 5
echo OUTER_R1120=$(cat /root/logs/p4251_r1120_lean_outer.pid)
echo OUTER_R1121=$(cat /root/logs/p4251_r1121_lean_outer.pid)
ok=0
for i in $(seq 1 90); do
  a=0; b=0
  if [[ -f /root/logs/r1120_train.pid ]]; then
    tp=$(cat /root/logs/r1120_train.pid); kill -0 "$tp" 2>/dev/null && a=1
  fi
  if [[ -f /root/logs/r1121_train.pid ]]; then
    tp=$(cat /root/logs/r1121_train.pid); kill -0 "$tp" 2>/dev/null && b=1
  fi
  echo "poll=$i r1120_alive=$a r1121_alive=$b"
  if [[ $a -eq 1 && $b -eq 1 ]]; then
    ok=1; break
  fi
  sleep 2
done
[[ $ok -eq 1 ]] || { echo FATAL trains not started; tail -80 /root/logs/r1120_lean_warm.log /root/logs/r1121_lean_warm.log /root/logs/p4251_r1120_lean_outer.nohup /root/logs/p4251_r1121_lean_outer.nohup 2>/dev/null || true; exit 1; }
echo TRAIN_R1120=$(cat /root/logs/r1120_train.pid)
echo TRAIN_R1121=$(cat /root/logs/r1121_train.pid)
cat /root/affine_data/r1120_train_launched.json
cat /root/affine_data/r1121_train_launched.json
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4251_r1096_97_refute_r1120_21_armed.done
REMOTE
