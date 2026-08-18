#!/usr/bin/env bash
# p3832 host→brave: tar-upload R780 + launch train on GPUs 0,1 + wait→merge
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=r780-tammy-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-lolr
SRC=$ROOT/experiments/$EXP
LOG=$ROOT/experiments/fleet-rent/logs/p3832_r780_launch.log
TAR=/tmp/r780_exp_p3832.tar.gz
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes -p 40127 root@18.118.83.97)
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3832-r780] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
log "pack $EXP"
tar czf "$TAR" -C "$ROOT/experiments" "$EXP"
ls -lh "$TAR" | tee -a "$LOG"
log "scp tar → brave"
lium scp brave-raven-a9 "$TAR" /tmp/r780_exp_p3832.tar.gz 2>&1 | tee -a "$LOG"
log "extract + launch"
lium exec brave-raven-a9 "bash -lc 'set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r780_exp_p3832.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus01_p3832.sh >/root/logs/p3832_r780_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3832_r780_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r780_train_then_merge_p3832.sh >/root/logs/p3832_r780_wait.nohup 2>&1 &
echo \$! >/root/logs/p3832_r780_wait.pid
echo LEAN=\$(cat /root/logs/p3832_r780_lean.outer.pid) WAIT=\$(cat /root/logs/p3832_r780_wait.pid)
'" 2>&1 | tee -a "$LOG"
for i in $(seq 1 24); do
  out=$(lium exec brave-raven-a9 'bash -lc "test -f /root/logs/r780_train.pid && cat /root/logs/r780_train.pid; tail -5 /root/logs/r780_lean_warm.log 2>/dev/null; nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader"' 2>&1 | tail -20)
  echo "poll=$i $out" | tee -a "$LOG"
  echo "$out" | grep -qE '^[0-9]+$' && break || true
  if echo "$out" | grep -q 'TRAIN_ARMED\|TRAIN launched'; then break; fi
  sleep 5
done
lium exec brave-raven-a9 'bash -lc "echo ===; cat /root/affine_data/r780_train_launched.json 2>/dev/null; echo ===; ps -p \$(cat /root/logs/r780_train.pid 2>/dev/null || echo 0) -o pid,etime,cmd 2>/dev/null | head -3; nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv"' 2>&1 | tee -a "$LOG"
log DONE
