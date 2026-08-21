#!/usr/bin/env bash
# p4262: R1110 REFUTE m=−0.006349 ~−0.62× → exact-PID reap r252 :8002 GPUs4,5 → R1138 UltraLoLR TRAIN
# Do not touch R1123 TRAIN GPUs6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1138-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.127.229.127
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
echo "[p4262] sync $EXP → mine-r252"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1138 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2" port="$3"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok" && echo "$cmd" | grep -q -- "--port $port"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone; fi
}
pid=$(cat /root/logs/vllm_chall_r1110.pid 2>/dev/null || echo 188903)
reap "$pid" r1110_merged 8002
# also clear any leftover :8002 listeners matching r1110
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1110_merged; then reap "$p" r1110_merged 8002; fi
done < <(ss -lptn 'sport = :8002' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "VRAM4+5=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
# free stale merges (keep active r1110 until after train starts; r1138 uses fresh /tmp)
rm -rf /tmp/r1002_merged /tmp/r1105_merged /tmp/r1102_merged 2>/dev/null || true
df -h /tmp / | head -5
# confirm R1123 still training
tp=$(cat /root/logs/r1123_train.pid 2>/dev/null || true)
if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then echo R1123_OK pid=$tp; else echo WARN R1123; fi
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1138-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1138-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r252_gpus45_p4262.sh >/root/logs/p4262_r1138_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4262_r1138_lean_outer.pid
ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1138_train.pid ]]; then
    tp=$(cat /root/logs/r1138_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1138_PID=$tp
      head -40 /root/logs/r1138_lean_warm.log || true
      cat /root/affine_data/r1138_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4262_r1138_lean_outer.nohup /root/logs/r1138_lean_warm.log 2>/dev/null || true; exit 1; }
# after train owns GPUs, drop idle r1110 merge to free /tmp
rm -rf /tmp/r1110_merged
df -h /tmp | head -2
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4262_r1110_refute_r1138_armed.done
echo R1138_TRAIN_OK
REMOTE
echo "[p4262] R1138 armed"
