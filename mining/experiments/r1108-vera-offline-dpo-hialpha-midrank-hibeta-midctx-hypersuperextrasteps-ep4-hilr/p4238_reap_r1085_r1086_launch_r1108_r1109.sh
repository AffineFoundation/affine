#!/usr/bin/env bash
# p4238 host-side: R1085+R1086 REFUTE → reap r339 :8002/:8003 → R1108+R1109 Hyper HiLR TRAIN
# Do not touch teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1108=r1108-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
E1109=r1109-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")

echo "[p4238] sync $E1108 + $E1109 → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1108 /root/mining_src/$E1109 /root/r1108 /root/r1109 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1108"/. "root@${HOST}:/root/mining_src/$E1108/"
"${SCP[@]}" -r "$ROOT/$E1109"/. "root@${HOST}:/root/mining_src/$E1109/"

echo "[p4238] exact-PID reap R1085 :8002 pid=40506 + R1086 :8003 pid=38042"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap_one() {
  local CHALL_PID=$1 tag=$2 merged=$3 port=$4
  if kill -0 "$CHALL_PID" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$CHALL_PID/cmdline 2>/dev/null || true)
    echo "reap pid=$CHALL_PID cmd=$cmd"
    if echo "$cmd" | grep -q "$merged" && echo "$cmd" | grep -q -- "--port $port"; then
      pids="$CHALL_PID $(pgrep -P $CHALL_PID 2>/dev/null || true)"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo "reaped $tag"
    else
      echo "FATAL pid $CHALL_PID is not $tag :$port — abort"; exit 2
    fi
  else
    echo "pid $CHALL_PID already gone — check :$port"
    ss -lptn "sport = :$port" || true
  fi
}
reap_one 40506 R1085 r1085_merged 8002
reap_one 38042 R1086 r1086_merged 8003

for pair in "4,5" "6,7"; do
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    echo "VRAM$pair used_mib=$used iter=$i"
    [[ "$used" -lt 8192 ]] && break
    sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { echo FATAL GPUs $pair still busy; exit 1; }
done

# Confirm TK still up
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

chmod +x /root/mining_src/r1108-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1109-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1108-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r339_gpus45_p4238.sh >/root/logs/p4238_r1108_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4238_r1108_lean_outer.pid
nohup bash /root/mining_src/r1109-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r339_gpus67_p4238.sh >/root/logs/p4238_r1109_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4238_r1109_lean_outer.pid
sleep 4

ok=0
for id in r1108 r1109; do
  for i in $(seq 1 60); do
    if [[ -f /root/logs/${id}_train.pid ]]; then
      tp=$(cat /root/logs/${id}_train.pid)
      if kill -0 "$tp" 2>/dev/null; then
        echo TRAIN_${id}_PID=$tp
        head -30 /root/logs/${id}_lean_warm.log || true
        cat /root/affine_data/${id}_train_launched.json || true
        ok=$((ok+1))
        break
      fi
    fi
    sleep 2
  done
done
[[ "$ok" -eq 2 ]] || { echo FATAL trains not started ok=$ok; tail -80 /root/logs/p4238_r1108_lean_outer.nohup /root/logs/p4238_r1109_lean_outer.nohup /root/logs/r1108_lean_warm.log /root/logs/r1109_lean_warm.log 2>/dev/null || true; exit 1; }
echo BOTH_TRAINS_OK
REMOTE
