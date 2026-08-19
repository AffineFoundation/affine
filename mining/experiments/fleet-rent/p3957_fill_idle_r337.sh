#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3957_fill_idle_r337.log
: >"$LOG"
log() { echo "[p3957-fill] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes)
HOST=86.38.182.67; PORT=20295

sync_and_launch() {
  local hypo="$1" gpus="$2" expdir="$3" startsh="$4" waitsh="$5" mergesh="$6"
  log "sync $hypo → $HOST:$PORT GPUs $gpus"
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "mkdir -p /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$expdir /root/$hypo /root/logs /root/affine_data"
  scp "${SSH_OPTS[@]}" -P "$PORT" \
    "$ROOT/experiments/$expdir/train_dpo.py" \
    "$ROOT/experiments/$expdir/dpo_duel_reason.jsonl" \
    "$ROOT/experiments/$expdir/$startsh" \
    "$ROOT/experiments/$expdir/$waitsh" \
    "$ROOT/experiments/$expdir/$mergesh" \
    "root@$HOST:/root/mining_src/$expdir/"
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "bash -s" <<EOR
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
  echo "[p3957-\$HYP] VRAM used_mib=\$used iter=\$i"
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
rm -f /root/logs/\${HYP}_merge_launched.p3957 /root/logs/\${HYP}_merge.done /root/logs/\${HYP}_train.done
: >/root/logs/\${HYP}_train.nohup
export CUDA_VISIBLE_DEVICES=\$GPUS
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2
export OUT=/root/\$HYP
export DATA=/root/\$HYP/dpo_duel_reason.jsonl
export LOG=/root/logs/\${HYP}_train.nohup
bash /root/mining_src/\$EXP/\$START
nohup bash /root/mining_src/\$EXP/\$WAIT >/root/logs/\${HYP}_wait_merge.p3957.nohup 2>&1 &
echo \$! >/root/logs/\${HYP}_wait_merge.p3957.pid
echo "[p3957-\$HYP] TRAIN pid=\$(cat /root/logs/\${HYP}_train.pid) wait=\$(cat /root/logs/\${HYP}_wait_merge.p3957.pid) gpus=\$GPUS"
EOR
}

log "START fill R337 idle 2–7 with R869–R871 ShortCtx HiRank Lo/Hi + MidCtx MidRank Lo"
sync_and_launch r869 2,3 \
  r869-marsplan-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr \
  start_r869.sh wait_r869_train_then_merge_p3957.sh lean_merge_r869_gpus23_p3957.sh
sync_and_launch r870 4,5 \
  r870-marsplan-offline-dpo-hialpha-hirank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr \
  start_r870.sh wait_r870_train_then_merge_p3957.sh lean_merge_r870_gpus45_p3957.sh
sync_and_launch r871 6,7 \
  r871-marsplan-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr \
  start_r871.sh wait_r871_train_then_merge_p3957.sh lean_merge_r871_gpus67_p3957.sh
log "ALL THREE TRAINS ARMED"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3957_fill_idle_r337.done"
