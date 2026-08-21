#!/usr/bin/env bash
# Host → blank mine-r1214 (brave-shark-4d): upload R1218 stack + bootstrap.
set -euo pipefail
ROOT=/home/const/subnet120
SRC=$ROOT/mining
EXP=r1218-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-hilr
HOST=216.48.189.107
PORT=19050
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" "root@$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes -P "$PORT")
echo "[upload-r1218] $(date -u +%Y-%m-%dT%H:%M:%SZ) → $HOST:$PORT"
test -f "$SRC/.env"
test -f "$SRC/experiments/$EXP/train_dpo.py"
test -s "$SRC/experiments/$EXP/dpo_duel_reason.jsonl"
"${SSH[@]}" 'mkdir -p /root/mining_src /root/logs /root/hf /root/affine_data /root/r1218'
# mine.env from mining/.env (HF_TOKEN)
"${SCP[@]}" "$SRC/.env" "root@$HOST:/root/mine.env"
# experiment bundle
tar -C "$SRC/experiments" -czf /tmp/${EXP}.tgz "$EXP"
"${SCP[@]}" "/tmp/${EXP}.tgz" "root@$HOST:/tmp/${EXP}.tgz"
# affine_pkg if present (for later n80)
if [[ -f /tmp/affine_pkg_p4086.tgz ]]; then
  "${SCP[@]}" /tmp/affine_pkg_p4086.tgz "root@$HOST:/tmp/affine_pkg.tgz" || true
fi
"${SSH[@]}" bash -s <<REMOTE
set -euo pipefail
cd /root/mining_src
tar -xzf /tmp/${EXP}.tgz
mkdir -p s4-h138-f43-tok-dpo-l2 s4-h1-sft affine_pkg
cp -f /root/mining_src/${EXP}/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/${EXP}/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /tmp/affine_pkg.tgz ]]; then
  tar -xzf /tmp/affine_pkg.tgz -C /root/mining_src/affine_pkg --strip-components=0 2>/dev/null \
    || tar -xzf /tmp/affine_pkg.tgz -C /root/mining_src 2>/dev/null || true
fi
chmod +x /root/mining_src/${EXP}/*.sh
test -s /root/mine.env
grep -q HF_TOKEN /root/mine.env
nohup bash /root/mining_src/${EXP}/bootstrap_r1218_p4326.sh >/root/logs/p4326_r1218_bootstrap.outer.nohup 2>&1 &
echo \$! >/root/logs/p4326_r1218_bootstrap.outer.pid
echo BOOTSTRAP_PID=\$(cat /root/logs/p4326_r1218_bootstrap.outer.pid)
sleep 3
head -20 /root/logs/bootstrap_r1218_p4326.log 2>/dev/null || head -20 /root/logs/p4326_r1218_bootstrap.outer.nohup || true
REMOTE
echo "[upload-r1218] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
