#!/usr/bin/env bash
# p4260: R1112+R1113 REFUTE → exact-PID reap r924 :8004/:8003 → R1131+R1132 TRAIN
# R1112 m=-0.001311 ~-0.12× → R1131 MidCtx LoRank MidLoβ Hyper HiLR (GPUs4,5)
# R1113 m=-0.000686 ~-0.07× → R1132 ShortCtx LoRank Loβ Hyper HiLR (GPUs1,3)
# Do not touch R1114 TRAIN GPUs6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP31=r1131-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr
EXP32=r1132-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

chmod +x "$ROOT/$EXP31"/*.sh "$ROOT/$EXP32"/*.sh

echo "[p4260] sync $EXP31 + $EXP32 → mine-r924"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP31 /root/mining_src/$EXP32 /root/r1131 /root/r1132 /root/logs /root/affine_data /root/tmp"
"${SCP[@]}" -r "$ROOT/$EXP31"/. "root@${HOST}:/root/mining_src/$EXP31/"
"${SCP[@]}" -r "$ROOT/$EXP32"/. "root@${HOST}:/root/mining_src/$EXP32/"

# also land decision artifacts locally mirrored on pod if missing
"${SCP[@]}" /tmp/r1112_decision_reign36_wvk7.json /tmp/r1113_decision_reign36_wvk7.json \
  "root@${HOST}:/root/affine_data/" 2>/dev/null || true

echo "[p4260] exact-PID reap R1112 :8004 + R1113 :8003; purge stale merges; launch R1131+R1132"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap_chall() {
  local label="$1" pidfile="$2" merge_token="$3" port="$4"
  local pid
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap $label pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$merge_token" && echo "$cmd" | grep -q -- "--port $port"; then
      pids="$pid"
      kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do
        pids="$pids $k"
        gkids=$(pgrep -P "$k" 2>/dev/null || true)
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
      echo "reaped $label"
    else
      echo "FATAL pid $pid is not $label :$port — abort"; exit 2
    fi
  else
    echo "$label pid gone — check :$port"
    ss -lptn "sport = :$port" || true
  fi
  # stop leftover sim if any (exact match)
  SIM=$(ps -eo pid=,args= | awk -v tok="$merge_token" '/run_sim_duel.py/ && $0 ~ tok && !/awk/ {print $1}')
  for p in $SIM; do
    echo "stop leftover sim pid=$p"
    kill "$p" 2>/dev/null || true
  done
}

reap_chall R1112 /root/logs/vllm_chall_r1112.pid r1112_merged 8004
reap_chall R1113 /root/logs/vllm_chall_r1113.pid r1113_merged 8003

# wait GPUs free
for pair in "4,5" "1,3"; do
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    echo "VRAM$pair=$used iter=$i"
    [[ "$used" -lt 8192 ]] && break
    sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { echo FATAL GPUs $pair still busy; exit 1; }
done

# purge stale merges (keep R1114 / active)
rm -rf /tmp/r1112_merged /tmp/r1113_merged
df -h / | head -2
echo "overlay free after purge: $(df -PB1 / | awk 'NR==2{print $4}')"

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# R1114 must stay
tp=$(cat /root/r1114/train.pid 2>/dev/null || true)
if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
  echo "R1114_TRAIN_OK pid=$tp"
else
  echo "WARN R1114 train pid missing (do not reclaim 6,7 here)"
fi

chmod +x /root/mining_src/r1131-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1132-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh

nohup bash /root/mining_src/r1131-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr/lean_train_r924_gpus45_p4260.sh \
  >/root/logs/p4260_r1131_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4260_r1131_lean_outer.pid
sleep 3
nohup bash /root/mining_src/r1132-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r924_gpus13_p4260.sh \
  >/root/logs/p4260_r1132_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4260_r1132_lean_outer.pid
sleep 8
echo "R1131_TRAIN_PID=$(cat /root/logs/r1131_train.pid 2>/dev/null || echo missing)"
echo "R1132_TRAIN_PID=$(cat /root/logs/r1132_train.pid 2>/dev/null || echo missing)"
tail -n 20 /root/logs/r1131_lean_warm.log 2>/dev/null || true
tail -n 20 /root/logs/r1132_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4260_r1112_r1113_refute_r1131_r1132_armed.done
REMOTE
echo "[p4260] R1131+R1132 armed"
