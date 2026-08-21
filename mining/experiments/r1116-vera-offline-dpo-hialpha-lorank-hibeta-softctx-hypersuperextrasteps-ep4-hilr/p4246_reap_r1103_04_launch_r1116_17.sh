#!/usr/bin/env bash
# p4246 host-side: R1103+R1104 REFUTE → reap crown :8004/:8003 → R1116+R1117 SoftCtx Hyper HiLR TRAIN
# Do not touch teacher:8000 GPU0 / king:8001 GPU2 / GPUs 6,7. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1116=r1116-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
E1117=r1117-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4246] sync $E1116 + $E1117 → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1116 /root/mining_src/$E1117 /root/r1116 /root/r1117 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1116"/. "root@${HOST}:/root/mining_src/$E1116/"
"${SCP[@]}" -r "$ROOT/$E1117"/. "root@${HOST}:/root/mining_src/$E1117/"

echo "[p4246] exact-PID reap R1103 :8004 pid=271664 + R1104 :8003 pid=270855"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail

reap_one() {
  local CHALL_PID=$1 TAG=$2 MERGED=$3 PORTN=$4
  if kill -0 "$CHALL_PID" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
    echo "reap pid=$CHALL_PID cmd=$cmd"
    if echo "$cmd" | grep -q "$MERGED" && echo "$cmd" | grep -q -- "--port $PORTN"; then
      pids="$CHALL_PID"
      kids=$(pgrep -P $CHALL_PID 2>/dev/null || true)
      for k in $kids; do
        pids="$pids $k"
        gkids=$(pgrep -P $k 2>/dev/null || true)
        for g in $gkids; do pids="$pids $g"; done
      done
      echo "kill set ($TAG): $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo "reaped $TAG"
    else
      echo "FATAL pid $CHALL_PID is not $TAG — abort"; exit 2
    fi
  else
    echo "pid $CHALL_PID already gone — check :$PORTN"
    ss -lptn "sport = :$PORTN" || true
  fi
}

reap_one 271664 R1103 r1103_merged 8004
reap_one 270855 R1104 r1104_merged 8003

for i in $(seq 1 90); do
  used13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM1,3=$used13 VRAM4,5=$used45 iter=$i"
  [[ "$used13" -lt 8192 && "$used45" -lt 8192 ]] && break
  sleep 2
done
used13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used13" -lt 8192 ]] || { echo FATAL GPUs 1,3 still busy; exit 1; }
[[ "$used45" -lt 8192 ]] || { echo FATAL GPUs 4,5 still busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1116-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1117-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1116-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_crown_gpus13_p4246.sh >/root/logs/p4246_r1116_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4246_r1116_lean_outer.pid
nohup bash /root/mining_src/r1117-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_crown_gpus45_p4246.sh >/root/logs/p4246_r1117_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4246_r1117_lean_outer.pid
sleep 6

ok=0
for i in $(seq 1 60); do
  a=0; b=0
  if [[ -f /root/logs/r1116_train.pid ]]; then
    tp=$(cat /root/logs/r1116_train.pid)
    if kill -0 "$tp" 2>/dev/null; then echo TRAIN_r1116_PID=$tp; a=1; fi
  fi
  if [[ -f /root/logs/r1117_train.pid ]]; then
    tp=$(cat /root/logs/r1117_train.pid)
    if kill -0 "$tp" 2>/dev/null; then echo TRAIN_r1117_PID=$tp; b=1; fi
  fi
  if [[ $a -eq 1 && $b -eq 1 ]]; then
    head -30 /root/logs/r1116_lean_warm.log || true
    head -30 /root/logs/r1117_lean_warm.log || true
    cat /root/affine_data/r1116_train_launched.json || true
    cat /root/affine_data/r1117_train_launched.json || true
    ok=1; break
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4246_r1116_lean_outer.nohup /root/logs/p4246_r1117_lean_outer.nohup /root/logs/r1116_lean_warm.log /root/logs/r1117_lean_warm.log 2>/dev/null || true; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4246_r1103_04_refute_r1116_17_armed.done
echo R1116_R1117_TRAIN_OK
REMOTE
