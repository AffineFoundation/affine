#!/usr/bin/env bash
# p4260b: R1116+R1117 REFUTE → exact-PID reap crown :8004/:8003 → R1133+R1134 TRAIN
# Do not touch R1129 TRAIN GPUs6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP33=r1133-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-hilr
EXP34=r1134-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP33"/*.sh "$ROOT/$EXP34"/*.sh
echo "[p4260b] sync → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP33 /root/mining_src/$EXP34 /root/r1133 /root/r1134 /root/logs /root/affine_data /root/tmp"
"${SCP[@]}" -r "$ROOT/$EXP33"/. "root@${HOST}:/root/mining_src/$EXP33/"
"${SCP[@]}" -r "$ROOT/$EXP34"/. "root@${HOST}:/root/mining_src/$EXP34/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap_chall() {
  local label="$1" pidfile="$2" merge_token="$3" port="$4"
  local pid; pid=$(cat "$pidfile" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap $label pid=$pid"
    if echo "$cmd" | grep -q "$merge_token" && echo "$cmd" | grep -q -- "--port $port"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo "reaped $label"
    else echo "FATAL wrong pid $pid"; exit 2; fi
  else echo "$label already gone"; fi
  SIM=$(ps -eo pid=,args= | awk -v tok="$merge_token" '/run_sim_duel.py/ && $0 ~ tok && !/awk/ {print $1}')
  for p in $SIM; do kill "$p" 2>/dev/null || true; done
}
reap_chall R1116 /root/logs/vllm_chall_r1116.pid r1116_merged 8004
reap_chall R1117 /root/logs/vllm_chall_r1117.pid r1117_merged 8003
for pair in "1,3" "4,5"; do
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    echo "VRAM$pair=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { echo FATAL busy $pair; exit 1; }
done
rm -rf /tmp/r1116_merged /tmp/r1117_merged
df -h / | head -2
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
tp=$(cat /root/r1129/train.pid 2>/dev/null || true)
if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then echo "R1129_TRAIN_OK pid=$tp"; else echo "WARN R1129 missing"; fi
chmod +x /root/mining_src/r1133-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
chmod +x /root/mining_src/r1134-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1133-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_crown_gpus13_p4260.sh >/root/logs/p4260_r1133_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4260_r1133_lean_outer.pid
sleep 3
nohup bash /root/mining_src/r1134-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_crown_gpus45_p4260.sh >/root/logs/p4260_r1134_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4260_r1134_lean_outer.pid
sleep 8
echo "R1133_TRAIN_PID=$(cat /root/logs/r1133_train.pid 2>/dev/null || echo missing)"
echo "R1134_TRAIN_PID=$(cat /root/logs/r1134_train.pid 2>/dev/null || echo missing)"
tail -n 15 /root/logs/r1133_lean_warm.log 2>/dev/null || true
tail -n 15 /root/logs/r1134_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4260_r1116_r1117_refute_r1133_r1134_armed.done
REMOTE
echo "[p4260b] R1133+R1134 armed"
