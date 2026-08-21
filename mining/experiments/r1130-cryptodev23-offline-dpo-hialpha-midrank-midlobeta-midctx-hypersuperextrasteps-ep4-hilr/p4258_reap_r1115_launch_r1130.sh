#!/usr/bin/env bash
# p4258 host-side: R1115 REFUTE → reap r926 :8002 GPUs3,4 → R1130 MidCtx MidRank MidLoβ Hyper HiLR TRAIN
# Do not touch teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1130=r1130-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=93.120.231.186
PORT=32301
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4258] sync $E1130 → mine-r926"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1130 /root/r1130 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1130"/. "root@${HOST}:/root/mining_src/$E1130/"

echo "[p4258] exact-PID reap R1115 :8002 pid=156721"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
CHALL_PID=156721
if kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$CHALL_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r1115_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
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
    echo "reaped R1115"
  else
    echo "FATAL pid $CHALL_PID is not R1115 :8002 — abort"; exit 2
  fi
else
  echo "pid $CHALL_PID already gone — check :8002"
  ss -lptn "sport = :8002" || true
fi

# stop leftover waiters that would re-arm R1115 n80
for pf in /root/logs/r1115_merge_then_n80.pid /root/logs/r1115_wait_merge.pid /root/logs/p4244_r1115_lean_outer.pid; do
  if [[ -f "$pf" ]]; then
    wp=$(cat "$pf" 2>/dev/null || true)
    if [[ -n "${wp:-}" ]] && kill -0 "$wp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$wp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -qE 'r1115|p4244'; then
        echo "stop leftover waiter pid=$wp"
        kill "$wp" 2>/dev/null || true
      fi
    fi
  fi
done

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "VRAM3,4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 3,4 still busy; nvidia-smi; exit 1; }

rm -rf /tmp/r1115_merged

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1130-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1130-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_h100_gpus34_p4258.sh >/root/logs/p4258_r1130_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4258_r1130_lean_outer.pid
sleep 4

ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1130_train.pid ]]; then
    tp=$(cat /root/logs/r1130_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1130_PID=$tp
      head -40 /root/logs/r1130_lean_warm.log || true
      cat /root/affine_data/r1130_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4258_r1130_lean_outer.nohup /root/logs/r1130_lean_warm.log 2>/dev/null || true; exit 1; }
echo R1130_TRAIN_OK
REMOTE
