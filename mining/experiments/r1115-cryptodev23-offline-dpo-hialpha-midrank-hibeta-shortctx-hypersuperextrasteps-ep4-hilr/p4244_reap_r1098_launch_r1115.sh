#!/usr/bin/env bash
# p4244 host-side: R1098 REFUTE → reap r926 :8002 GPUs3,4 → R1115 ShortCtx MidRank Hiβ Hyper HiLR TRAIN
# Do not touch teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1115=r1115-cryptodev23-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=93.120.231.186
PORT=32301
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4244] sync $E1115 → mine-r926"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1115 /root/r1115 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1115"/. "root@${HOST}:/root/mining_src/$E1115/"

echo "[p4244] exact-PID reap R1098 :8002 pid=152268"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=152268
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1098_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1098"
  else
    echo "FATAL pid $CHALL_PID is not R1098 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi

# also stop leftover sim pidfile if any
if [[ -f /root/logs/r1098_sim_wvk7.pid ]]; then
  sp=$(cat /root/logs/r1098_sim_wvk7.pid 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    echo "sim still alive $sp — leave (n80 already done)"
  fi
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "VRAM3,4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 3,4 still busy; nvidia-smi; exit 1; }

# free old merge disk (train will rebuild)
rm -rf /tmp/r1098_merged

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1115-cryptodev23-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1115-cryptodev23-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_h100_gpus34_p4244.sh >/root/logs/p4244_r1115_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4244_r1115_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1115_train.pid ]]; then
    tp=$(cat /root/logs/r1115_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1115_PID=$tp
      head -40 /root/logs/r1115_lean_warm.log || true
      cat /root/affine_data/r1115_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4244_r1115_lean_outer.nohup /root/logs/r1115_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1115_TRAIN_OK
REMOTE
