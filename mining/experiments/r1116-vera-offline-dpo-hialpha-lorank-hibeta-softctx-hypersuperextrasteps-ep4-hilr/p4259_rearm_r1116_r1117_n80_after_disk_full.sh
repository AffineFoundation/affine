#!/usr/bin/env bash
# p4259: R1116+R1117 MERGE done @04:28Z but both chall vLLM failed with
# OSError Errno 28 (overlay /tmp full of ~48 stale *66G merges). Freed disk,
# re-arm v4 n80s on crown GPUs 1,3 (:8004) and 4,5 (:8003). TMPDIR→/root/tmp.
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1129 GPUs6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP16=r1116-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
EXP17=r1117-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

echo "[p4259] sync lean scripts → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP16 /root/mining_src/$EXP17 /root/logs /root/affine_data /root/tmp"
"${SCP[@]}" -r "$ROOT/$EXP16"/. "root@${HOST}:/root/mining_src/$EXP16/"
"${SCP[@]}" -r "$ROOT/$EXP17"/. "root@${HOST}:/root/mining_src/$EXP17/"

"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
EXP16=r1116-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-hilr
EXP17=r1117-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-hilr
export TMPDIR=/root/tmp
mkdir -p /root/tmp /root/logs

avail=$(df -PB1 / | awk 'NR==2{print $4}')
echo "overlay_avail_bytes=$avail"
[[ "$avail" -gt 200000000000 ]] || { echo FATAL disk still tight; df -h /; exit 1; }

n16=$(ls /tmp/r1116_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
n17=$(ls /tmp/r1117_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f /tmp/r1116_merged/config.json && "${n16:-0}" -ge 16 ]] || { echo FATAL r1116 merge; exit 1; }
[[ -f /tmp/r1117_merged/config.json && "${n17:-0}" -ge 16 ]] || { echo FATAL r1117 merge; exit 1; }
echo "shards r1116=$n16 r1117=$n17"

used13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
echo "VRAM1+3=$used13 VRAM4+5=$used45"
[[ "$used13" -lt 8192 ]] || { echo FATAL GPUs 1,3 busy; exit 1; }
[[ "$used45" -lt 8192 ]] || { echo FATAL GPUs 4,5 busy; exit 1; }

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK

# R1129 train must stay up on 6,7
tp=$(cat /root/r1129/train.pid 2>/dev/null || true)
if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
  echo "R1129_TRAIN_OK pid=$tp"
else
  echo "WARN R1129 train pid missing (do not reclaim 6,7 here)"
fi

for port in 8003 8004; do
  if curl -sf -m 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "FATAL :$port already serving"; exit 1
  fi
done

# Reap any dead chall leftovers by exact pid only (never pkill -f)
for pf in /root/logs/vllm_chall_r1116.pid /root/logs/vllm_chall_r1117.pid; do
  pid=$(cat "$pf" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "kill leftover chall pid=$pid from $pf"
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
done

chmod +x /root/mining_src/$EXP16/lean_chall_n80_crown_gpus13_p4246.sh
chmod +x /root/mining_src/$EXP17/lean_chall_n80_crown_gpus45_p4246.sh

# Launch R1116 first; brief settle; then R1117
nohup env TMPDIR=/root/tmp bash /root/mining_src/$EXP16/lean_chall_n80_crown_gpus13_p4246.sh \
  >/root/logs/p4259_r1116_chall_n80_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4259_r1116_chall_n80_outer.pid
sleep 20
nohup env TMPDIR=/root/tmp bash /root/mining_src/$EXP17/lean_chall_n80_crown_gpus45_p4246.sh \
  >/root/logs/p4259_r1117_chall_n80_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4259_r1117_chall_n80_outer.pid

sleep 25
echo "OUTER16=$(cat /root/logs/p4259_r1116_chall_n80_outer.pid)"
echo "OUTER17=$(cat /root/logs/p4259_r1117_chall_n80_outer.pid)"
echo "CHALL16=$(cat /root/logs/vllm_chall_r1116.pid 2>/dev/null || echo pending)"
echo "CHALL17=$(cat /root/logs/vllm_chall_r1117.pid 2>/dev/null || echo pending)"
tail -n 25 /root/logs/p4246_r1116_chall_n80_wvk7.log 2>/dev/null || true
tail -n 25 /root/logs/p4246_r1117_chall_n80_wvk7.log 2>/dev/null || true
tail -n 15 /root/logs/p4259_r1116_chall_n80_outer.nohup 2>/dev/null || true
tail -n 15 /root/logs/p4259_r1117_chall_n80_outer.nohup 2>/dev/null || true
df -h / | tail -1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4259_r1116_r1117_n80_armed.done
REMOTE
echo "[p4259] R1116+R1117 n80 re-armed after disk cleanup"
