#!/usr/bin/env bash
# p4268: R1128 MERGE done but lean_chall stuck polling wrong GPUs (1,2 vs 6,7).
# Exact-PID reap stuck lean; sync fixed script; re-arm chall+n80 on GPUs 6,7.
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1141 / R1142.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4268] sync fixed lean_chall → mine-r340"
"${SCP[@]}" "$ROOT/$EXP/lean_chall_n80_r340_gpus67_p4255.sh" "root@${HOST}:/root/mining_src/$EXP/lean_chall_n80_r340_gpus67_p4255.sh"

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
echo "=== pre GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# Exact-PID reap stuck lean_chall + its sleep child; leave wait parent to exit
LEAN_PID=$(ps -eo pid=,args= | awk '/lean_chall_n80_r340_gpus67_p4255\.sh/ && !/awk/ {print $1; exit}')
if [[ -n "${LEAN_PID:-}" ]]; then
  echo "reap lean pid=$LEAN_PID"
  # children first
  for c in $(ps --ppid "$LEAN_PID" -o pid= 2>/dev/null || true); do
    echo "  reap child $c"; kill "$c" 2>/dev/null || true
  done
  kill "$LEAN_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$LEAN_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$LEAN_PID" 2>/dev/null || true
fi
# If old wait_merge_then_n80 still holding, leave it; we nohup lean directly
sleep 2
# Confirm merge still present
[[ -f /tmp/r1128_merged/config.json ]] || { echo FATAL no merge; exit 1; }
n=$(ls /tmp/r1128_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
echo "merge_shards=$n"
[[ "$n" -ge 16 ]] || { echo FATAL incomplete merge; exit 1; }
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
echo "VRAM6+7=$used"
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# Confirm R1141/R1142 still training
ps -eo pid,args | grep -E 'r1141/train|r1142/train' | grep -v grep | head -5 || true
chmod +x /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_chall_n80_r340_gpus67_p4255.sh
# Verify fix landed
grep -n 'nvidia-smi -i 6,7' /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_chall_n80_r340_gpus67_p4255.sh | head -3
nohup bash /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_chall_n80_r340_gpus67_p4255.sh \
  >/root/logs/p4268_r1128_lean_chall_rearm.nohup 2>&1 &
echo $! | tee /root/logs/p4268_r1128_lean_chall_rearm.pid
sleep 25
echo "=== post ==="
tail -n 40 /root/logs/p4255_r1128_chall_n80_wvk7.log 2>/dev/null || true
tail -n 20 /root/logs/p4268_r1128_lean_chall_rearm.nohup 2>/dev/null || true
ps -eo pid,etime,args | grep -E 'lean_chall_n80_r340_gpus67|vllm.*r1128' | grep -v grep | head -10
ss -ltnp | grep -E ':800[0-9]' || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4268_r1128_n80_rearmed.done
REMOTE
echo "[p4268] R1128 n80 re-armed"
