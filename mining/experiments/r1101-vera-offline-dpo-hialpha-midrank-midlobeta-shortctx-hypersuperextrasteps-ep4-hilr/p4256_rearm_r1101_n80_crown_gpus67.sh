#!/usr/bin/env bash
# p4256 host-side: R1101 merge was DONE since 02:23Z but n80 waiter died on a
# wrong lean path (lean_chall_n80_r338_gpus45…). GPUs 6,7 idle on mine-crown-1.
# Re-arm v4 n80 with the correct crown GPUs6,7 lean. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1101-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4256] sync $EXP → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
EXP=r1101-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
# confirm merge intact + GPUs 6,7 idle + TK warm; do not touch R1116/R1117
n=$(ls /tmp/r1101_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f /tmp/r1101_merged/config.json ]] || { echo FATAL no merge; exit 1; }
[[ "${n:-0}" -ge 16 ]] || { echo FATAL shards=$n; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
echo "VRAM6+7=$used shards=$n"
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs 6,7 busy; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# ensure sibling trains still alive
ps -eo pid,args | grep -E 'r1116/train|r1117/train' | grep -v grep | head -5 || true
# no live chall on :8002
if curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
  echo FATAL :8002 already serving; exit 1
fi
chmod +x /root/mining_src/$EXP/lean_chall_n80_crown_r1101_gpus67_p4232.sh
chmod +x /root/mining_src/$EXP/wait_r1101_merge_then_n80_p4232.sh
nohup bash /root/mining_src/$EXP/lean_chall_n80_crown_r1101_gpus67_p4232.sh \
  >/root/logs/p4256_r1101_chall_n80_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4256_r1101_chall_n80_outer.pid
sleep 12
echo "OUTER_PID=$(cat /root/logs/p4256_r1101_chall_n80_outer.pid)"
echo "CHALL_PID=$(cat /root/logs/vllm_chall_r1101.pid 2>/dev/null || echo pending)"
tail -n 40 /root/logs/p4232_r1101_chall_n80_wvk7.log 2>/dev/null || true
tail -n 20 /root/logs/p4256_r1101_chall_n80_outer.nohup 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4256_r1101_n80_armed.done
REMOTE
echo "[p4256] R1101 n80 re-armed on crown GPUs6,7"
