#!/usr/bin/env bash
# p4254 host-side: R1108+R1109 REFUTE → reap r339 :8002 GPUs4,5 + :8003 GPUs6,7 → R1126+R1127 TRAIN
# Do not touch teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP26=r1126-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-hilr
EXP27=r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4254] sync $EXP26 + $EXP27 → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP26 /root/mining_src/$EXP27 /root/r1126 /root/r1127 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP26"/. "root@${HOST}:/root/mining_src/$EXP26/"
"${SCP[@]}" -r "$ROOT/$EXP27"/. "root@${HOST}:/root/mining_src/$EXP27/"

# pull decision artifacts
mkdir -p "$ROOT/r1108-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-hilr" \
         "$ROOT/r1109-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr"
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1108_sim_result_reign36_wvk7.json" \
  "$ROOT/r1108-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-hilr/r1108_decision_reign36_wvk7.json" || true
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1109_sim_result_reign36_wvk7.json" \
  "$ROOT/r1109-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr/r1109_decision_reign36_wvk7.json" || true

echo "[p4254] exact-PID reap R1108 :8002 pid=47735 + R1109 :8003 pid=45825"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap_one() {
  local CHALL_PID=$1 EXPECT_MERGE=$2 EXPECT_PORT=$3 LABEL=$4
  if kill -0 "$CHALL_PID" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
    echo "reap pid=$CHALL_PID cmd=$cmd"
    if echo "$cmd" | grep -q "$EXPECT_MERGE" && echo "$cmd" | grep -q -- "--port $EXPECT_PORT"; then
      pids="$CHALL_PID"
      kids=$(pgrep -P $CHALL_PID 2>/dev/null || true)
      for k in $kids; do
        pids="$pids $k"
        gkids=$(pgrep -P $k 2>/dev/null || true)
        for g in $gkids; do pids="$pids $g"; done
      done
      echo "kill set ($LABEL): $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo "reaped $LABEL"
    else
      echo "FATAL pid $CHALL_PID is not $LABEL :$EXPECT_PORT — abort"; exit 2
    fi
  else
    echo "pid $CHALL_PID already gone — check :$EXPECT_PORT"
    ss -lptn "sport = :$EXPECT_PORT" || true
  fi
}
reap_one 47735 r1108_merged 8002 R1108
reap_one 45825 r1109_merged 8003 R1109

for pair in "4,5" "6,7"; do
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    echo "VRAM$pair=$used iter=$i"
    [[ "$used" -lt 8192 ]] && break
    sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { echo FATAL GPUs $pair still busy; exit 1; }
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1126-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1126-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r339_gpus45_p4254.sh >/root/logs/p4254_r1126_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4254_r1126_lean_outer.pid
nohup bash /root/mining_src/r1127-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r339_gpus67_p4254.sh >/root/logs/p4254_r1127_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4254_r1127_lean_outer.pid

sleep 8
echo "R1126_TRAIN_PID=$(cat /root/logs/r1126_train.pid 2>/dev/null || echo missing)"
echo "R1127_TRAIN_PID=$(cat /root/logs/r1127_train.pid 2>/dev/null || echo missing)"
tail -n 20 /root/logs/r1126_lean_warm.log 2>/dev/null || true
tail -n 20 /root/logs/r1127_lean_warm.log 2>/dev/null || true
cat /root/affine_data/r1126_train_launched.json 2>/dev/null || true
cat /root/affine_data/r1127_train_launched.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4254_r1108_r1109_refute_r1126_r1127_armed.done
REMOTE
echo "[p4254] R1126+R1127 armed"
