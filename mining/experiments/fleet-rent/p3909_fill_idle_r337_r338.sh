#!/usr/bin/env bash
# p3909: fill idle GPUs 2-5 on R337/R338 — R796 reclaim + R830/R831/R832 Soft Mid Mid Soft transfers
set -euo pipefail
ROOT=/home/const/subnet120/mining
source /home/const/subnet120/.venv/bin/activate
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes"

sync_and_launch() {
  local host="$1" port="$2" hypo="$3" gpus="$4" expdir="$5" startsh="$6"
  echo "[p3909] sync $hypo → $host:$port GPUs $gpus"
  ssh $SSH_OPTS -p "$port" "root@$host" \
    "mkdir -p /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$expdir /root/$hypo /root/logs /root/affine_data"
  scp $SSH_OPTS -P "$port" \
    "$ROOT/experiments/$expdir/train_dpo.py" \
    "$ROOT/experiments/$expdir/dpo_duel_reason.jsonl" \
    "$ROOT/experiments/$expdir/$startsh" \
    "root@$host:/root/mining_src/$expdir/"
  ssh $SSH_OPTS -p "$port" "root@$host" "bash -s" <<EOS
set -euo pipefail
EXP='$expdir'
HYP='$hypo'
GPUS='$gpus'
START='$startsh'
cp -f /root/mining_src/\$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/\$EXP/dpo_duel_reason.jsonl /root/\$HYP/dpo_duel_reason.jsonl
chmod +x /root/mining_src/\$EXP/*.sh
for i in \$(seq 1 30); do
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$GPUS | awk '{s+=\$1} END{print s+0}')
  echo "[p3909-\$HYP] VRAM used_mib=\$used iter=\$i"
  [[ "\$used" -lt 8192 ]] && break
  sleep 2
done
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$GPUS | awk '{s+=\$1} END{print s+0}')
[[ "\$used" -lt 8192 ]] || { echo FATAL GPUs \$GPUS busy; exit 1; }
if [[ -f /root/logs/\${HYP}_train.pid ]]; then
  old=\$(cat /root/logs/\${HYP}_train.pid || true)
  if [[ "\$old" =~ ^[0-9]+\$ ]] && kill -0 "\$old" 2>/dev/null; then
    echo FATAL already alive \$old; exit 1
  fi
fi
rm -rf /root/\$HYP/train
mkdir -p /root/\$HYP/train
: >/root/logs/\${HYP}_train.nohup
export CUDA_VISIBLE_DEVICES=\$GPUS
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2
export OUT=/root/\$HYP
export DATA=/root/\$HYP/dpo_duel_reason.jsonl
export LOG=/root/logs/\${HYP}_train.nohup
bash /root/mining_src/\$EXP/\$START
echo "[p3909-\$HYP] TRAIN pid=\$(cat /root/logs/\${HYP}_train.pid) gpus=\$GPUS"
EOS
}

# R337: R796 on 2,3 + R830 on 4,5
sync_and_launch 86.38.182.67 20295 r796 2,3 \
  r796-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r796.sh

sync_and_launch 86.38.182.67 20295 r830 4,5 \
  r830-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r830.sh

# R338: R831 on 2,3 + R832 on 4,5
sync_and_launch 86.38.182.55 20299 r831 2,3 \
  r831-marsplan-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr \
  start_r831.sh

sync_and_launch 86.38.182.55 20299 r832 4,5 \
  r832-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r832.sh

echo "[p3909] ALL FOUR TRAINS ARMED"
