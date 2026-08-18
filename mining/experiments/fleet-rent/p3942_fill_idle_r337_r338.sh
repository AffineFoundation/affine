#!/usr/bin/env bash
# p3942: fill idle GPUs 2–7 on R337/R338 with R852–R857 marsplan Soft Mid Mid Soft UltraLoLR axes
# Never pkill -f. Keep teacher on 0,1. Arm wait→merge only (host-relay later).
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3942_fill_idle_r337_r338.log
: >"$LOG"
log() { echo "[p3942-fill] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes)

sync_and_launch() {
  local host="$1" port="$2" hypo="$3" gpus="$4" expdir="$5" startsh="$6" waitsh="$7" mergesh="$8"
  log "sync $hypo → $host:$port GPUs $gpus"
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
    "mkdir -p /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$expdir /root/$hypo /root/logs /root/affine_data"
  scp "${SSH_OPTS[@]}" -P "$port" \
    "$ROOT/experiments/$expdir/train_dpo.py" \
    "$ROOT/experiments/$expdir/dpo_duel_reason.jsonl" \
    "$ROOT/experiments/$expdir/$startsh" \
    "$ROOT/experiments/$expdir/$waitsh" \
    "$ROOT/experiments/$expdir/$mergesh" \
    "root@$host:/root/mining_src/$expdir/"
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" "bash -s" <<EOS
set -euo pipefail
EXP='$expdir'
HYP='$hypo'
GPUS='$gpus'
START='$startsh'
WAIT='$waitsh'
cp -f /root/mining_src/\$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/\$EXP/dpo_duel_reason.jsonl /root/\$HYP/dpo_duel_reason.jsonl
chmod +x /root/mining_src/\$EXP/*.sh
for i in \$(seq 1 30); do
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$GPUS | awk '{s+=\$1} END{print s+0}')
  echo "[p3942-\$HYP] VRAM used_mib=\$used iter=\$i"
  [[ "\$used" -lt 8192 ]] && break
  sleep 2
done
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$GPUS | awk '{s+=\$1} END{print s+0}')
[[ "\$used" -lt 8192 ]] || { echo FATAL GPUs \$GPUS busy used=\$used; exit 1; }
if [[ -f /root/logs/\${HYP}_train.pid ]]; then
  old=\$(cat /root/logs/\${HYP}_train.pid || true)
  if [[ "\$old" =~ ^[0-9]+\$ ]] && kill -0 "\$old" 2>/dev/null; then
    echo FATAL already alive \$old; exit 1
  fi
fi
rm -rf /root/\$HYP/train
mkdir -p /root/\$HYP/train
rm -f /root/logs/\${HYP}_merge_launched.p3942 /root/logs/\${HYP}_merge.done /root/logs/\${HYP}_train.done
: >/root/logs/\${HYP}_train.nohup
export CUDA_VISIBLE_DEVICES=\$GPUS
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2
export OUT=/root/\$HYP
export DATA=/root/\$HYP/dpo_duel_reason.jsonl
export LOG=/root/logs/\${HYP}_train.nohup
bash /root/mining_src/\$EXP/\$START
nohup bash /root/mining_src/\$EXP/\$WAIT >/root/logs/\${HYP}_wait_merge.p3942.nohup 2>&1 &
echo \$! >/root/logs/\${HYP}_wait_merge.p3942.pid
echo "[p3942-\$HYP] TRAIN pid=\$(cat /root/logs/\${HYP}_train.pid) wait=\$(cat /root/logs/\${HYP}_wait_merge.p3942.pid) gpus=\$GPUS"
EOS
}

log "START fill R337+R338 idle 2–7"

# R337 gentle-shark
sync_and_launch 86.38.182.67 20295 r852 2,3 \
  r852-marsplan-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r852.sh wait_r852_train_then_merge_p3942.sh lean_merge_r852_gpus23_p3942.sh

sync_and_launch 86.38.182.67 20295 r853 4,5 \
  r853-marsplan-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r853.sh wait_r853_train_then_merge_p3942.sh lean_merge_r853_gpus45_p3942.sh

sync_and_launch 86.38.182.67 20295 r854 6,7 \
  r854-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r854.sh wait_r854_train_then_merge_p3942.sh lean_merge_r854_gpus67_p3942.sh

# R338 calm-lion
sync_and_launch 86.38.182.55 20299 r855 2,3 \
  r855-marsplan-offline-dpo-hialpha-hirank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr \
  start_r855.sh wait_r855_train_then_merge_p3942.sh lean_merge_r855_gpus23_p3942.sh

sync_and_launch 86.38.182.55 20299 r856 4,5 \
  r856-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r856.sh wait_r856_train_then_merge_p3942.sh lean_merge_r856_gpus45_p3942.sh

sync_and_launch 86.38.182.55 20299 r857 6,7 \
  r857-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r857.sh wait_r857_train_then_merge_p3942.sh lean_merge_r857_gpus67_p3942.sh

log "ALL SIX TRAINS ARMED"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3942_fill_idle_r337_r338.done"
