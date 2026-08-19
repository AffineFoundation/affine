#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3954_fill_idle_r338.log
: >"$LOG"
log() { echo "[p3954-fill] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes)
HOST=86.38.182.55; PORT=20299

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
  echo "[p3954-\$HYP] VRAM used_mib=\$used iter=\$i"
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
rm -f /root/logs/\${HYP}_merge_launched.p3954 /root/logs/\${HYP}_merge.done /root/logs/\${HYP}_train.done
: >/root/logs/\${HYP}_train.nohup
export CUDA_VISIBLE_DEVICES=\$GPUS
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2
export OUT=/root/\$HYP
export DATA=/root/\$HYP/dpo_duel_reason.jsonl
export LOG=/root/logs/\${HYP}_train.nohup
bash /root/mining_src/\$EXP/\$START
nohup bash /root/mining_src/\$EXP/\$WAIT >/root/logs/\${HYP}_wait_merge.p3954.nohup 2>&1 &
echo \$! >/root/logs/\${HYP}_wait_merge.p3954.pid
echo "[p3954-\$HYP] TRAIN pid=\$(cat /root/logs/\${HYP}_train.pid) wait=\$(cat /root/logs/\${HYP}_wait_merge.p3954.pid) gpus=\$GPUS"
EOR
}

log "START fill R338 idle 2–7 with R863–R865 MidRank SoftCtx"
sync_and_launch r863 2,3 \
  r863-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r863.sh wait_r863_train_then_merge_p3954.sh lean_merge_r863_gpus23_p3954.sh
sync_and_launch r864 4,5 \
  r864-marsplan-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r864.sh wait_r864_train_then_merge_p3954.sh lean_merge_r864_gpus45_p3954.sh
sync_and_launch r865 6,7 \
  r865-marsplan-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr \
  start_r865.sh wait_r865_train_then_merge_p3954.sh lean_merge_r865_gpus67_p3954.sh
log "ALL THREE TRAINS ARMED"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3954_fill_idle_r338.done"
