#!/usr/bin/env bash
# p4255 host-side: fill idle r340 GPUs 6,7 with R1128 SoftCtx LoRank MidLoβ Hyper HiLR TRAIN
# Do not touch teacher:8000 / king:8001 / R1120 / R1121. Never pkill -f.
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

echo "[p4255] sync $EXP → mine-r340"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1128 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
# confirm GPUs 6,7 idle
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
echo "VRAM6+7=$used"
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# ensure R1120/R1121 still training (do not disturb)
ps -eo pid,args | grep -E 'r1120/train|r1121/train' | grep -v grep | head -5 || true
chmod +x /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/*.sh
nohup bash /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/lean_train_r340_gpus67_p4255.sh >/root/logs/p4255_r1128_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4255_r1128_lean_outer.pid
sleep 10
echo "R1128_TRAIN_PID=$(cat /root/logs/r1128_train.pid 2>/dev/null || echo missing)"
tail -n 30 /root/logs/r1128_lean_warm.log 2>/dev/null || true
cat /root/affine_data/r1128_train_launched.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4255_r1128_armed.done
REMOTE
echo "[p4255] R1128 armed"
