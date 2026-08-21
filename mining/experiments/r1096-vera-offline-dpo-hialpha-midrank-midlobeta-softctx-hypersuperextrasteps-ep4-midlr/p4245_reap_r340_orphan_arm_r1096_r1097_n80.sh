#!/usr/bin/env bash
# p4245 host-side: R1096+R1097 merges sat idle (lean_chall was a stub).
# Exact-PID reap aborted R340 chall on CUDA 4,5 (pid 24856) — do not touch king:8001 / teacher:8000.
# Then arm real lean_chall+v4 n80 for R1096 (GPUs1,2 :8002) and R1097 (GPUs3,4 :8003).
# Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1096=r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr
E1097=r1097-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4245] sync $E1096 + $E1097 → mine-r340"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1096 /root/mining_src/$E1097 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1096"/. "root@${HOST}:/root/mining_src/$E1096/"
"${SCP[@]}" -r "$ROOT/$E1097"/. "root@${HOST}:/root/mining_src/$E1097/"

echo "[p4245] exact-PID reap aborted R340 chall pid=24856 (CUDA 4,5 overlap with king)"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
R340_PID=24856
if kill -0 "$R340_PID" 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/$R340_PID/cmdline 2>/dev/null || true)
  echo "reap pid=$R340_PID cmd=$cmd"
  if echo "$cmd" | grep -q "r340_merged" && echo "$cmd" | grep -q -- "--port 8002"; then
    pids="$R340_PID"
    # EngineCore children often under a different parent; gather by CUDA 4,5 workers that belong to this serve tree
    kids=$(pgrep -P $R340_PID 2>/dev/null || true)
    for k in $kids; do
      pids="$pids $k"
      gkids=$(pgrep -P $k 2>/dev/null || true)
      for g in $gkids; do pids="$pids $g"; done
    done
    # Also reap known orphan workers 25520/25521 if their cmdline is VLLM::Worker and GPU4/5
    for wp in 25520 25521 25191; do
      if kill -0 "$wp" 2>/dev/null; then
        wcmd=$(tr "\0" " " < /proc/$wp/cmdline 2>/dev/null || true)
        if echo "$wcmd" | grep -qE 'VLLM::Worker|VLLM::EngineCore'; then
          # only if parent chain leads to r340 or EngineCore under r340 tree
          pp=$(awk '/PPid/{print $2}' /proc/$wp/status 2>/dev/null || true)
          if [[ "$pp" == "$R340_PID" ]] || echo " $pids " | grep -q " $pp " || [[ "$wp" == "25191" ]]; then
            pids="$pids $wp"
          fi
        fi
      fi
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
    echo "reaped R340 orphan chall"
  else
    echo "FATAL pid $R340_PID is not r340_merged :8002 — abort"; exit 2
  fi
else
  echo "pid $R340_PID already gone"
fi

# Wait GPU4 free (king stays on GPU5)
for i in $(seq 1 60); do
  used4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | awk '{print $1+0}')
  echo "GPU4 used_mib=$used4 iter=$i"
  [[ "$used4" -lt 2048 ]] && break
  sleep 2
done
used4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | awk '{print $1+0}')
[[ "$used4" -lt 2048 ]] || { echo FATAL GPU4 still busy; nvidia-smi; exit 1; }

# Confirm GPUs 1-4 free enough
for gi in 1 2 3 4; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" | awk '{print $1+0}')
  echo "GPU$gi used_mib=$used"
  [[ "$used" -lt 2048 ]] || { echo FATAL GPU$gi busy; exit 1; }
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
[[ -f /tmp/r1096_merged/config.json ]] && echo R1096_MERGE_OK
[[ -f /tmp/r1097_merged/config.json ]] && echo R1097_MERGE_OK

chmod +x /root/mining_src/r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/*.sh
chmod +x /root/mining_src/r1097-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh

nohup bash /root/mining_src/r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/lean_chall_n80_r340_gpus12_p4226.sh >/root/logs/p4245_r1096_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4245_r1096_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r1097-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_chall_n80_r340_gpus34_p4226.sh >/root/logs/p4245_r1097_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4245_r1097_lean_outer.pid
sleep 6
echo "r1096_outer=$(cat /root/logs/p4245_r1096_lean_outer.pid) alive=$(kill -0 $(cat /root/logs/p4245_r1096_lean_outer.pid) 2>/dev/null && echo y || echo n)"
echo "r1097_outer=$(cat /root/logs/p4245_r1097_lean_outer.pid) alive=$(kill -0 $(cat /root/logs/p4245_r1097_lean_outer.pid) 2>/dev/null && echo y || echo n)"
tail -20 /root/logs/p4245_r1096_lean_outer.nohup || true
tail -20 /root/logs/p4245_r1097_lean_outer.nohup || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE

echo "[p4245] DONE arm R1096+R1097 n80 on mine-r340"
