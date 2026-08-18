#!/usr/bin/env bash
# p3833 host→brave: tar-upload R781 + launch train on GPUs 2,3 + wait→merge
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=r781-tammy-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-lolr
SRC=$ROOT/experiments/$EXP
LOG=$ROOT/experiments/fleet-rent/logs/p3833_r781_launch.log
TAR=/tmp/r781_exp_p3833.tar.gz
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3833-r781] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
log "pack $EXP"
tar czf "$TAR" -C "$ROOT/experiments" "$EXP"
ls -lh "$TAR" | tee -a "$LOG"
log "scp tar → brave"
lium scp brave-raven-a9 "$TAR" /tmp/r781_exp_p3833.tar.gz 2>&1 | tee -a "$LOG"
log "extract + launch"
lium exec brave-raven-a9 "bash -lc 'set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r781_exp_p3833.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus23_p3833.sh >/root/logs/p3833_r781_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3833_r781_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r781_train_then_merge_p3833.sh >/root/logs/p3833_r781_wait.nohup 2>&1 &
echo \$! >/root/logs/p3833_r781_wait.pid
echo LEAN=\$(cat /root/logs/p3833_r781_lean.outer.pid) WAIT=\$(cat /root/logs/p3833_r781_wait.pid)
'" 2>&1 | tee -a "$LOG"
for i in $(seq 1 36); do
  out=$(lium exec brave-raven-a9 'bash -lc "test -f /root/logs/r781_train.pid && cat /root/logs/r781_train.pid; tail -8 /root/logs/r781_lean_warm.log 2>/dev/null; nvidia-smi -i 2,3 --query-gpu=memory.used --format=csv,noheader"' 2>&1 | tail -25)
  echo "poll=$i $out" | tee -a "$LOG"
  if echo "$out" | grep -qE 'TRAIN_ARMED|TRAIN launched'; then break; fi
  if echo "$out" | grep -qE '^[0-9]+$' && echo "$out" | grep -vq '0 MiB'; then
    # pid present
    :
  fi
  # success if pid file exists and process alive
  alive=$(lium exec brave-raven-a9 'bash -lc "pid=$(cat /root/logs/r781_train.pid 2>/dev/null || echo); [[ -n $pid ]] && kill -0 $pid 2>/dev/null && echo ALIVE:$pid || echo DEAD"' 2>&1 | tail -3)
  echo "alive=$alive" | tee -a "$LOG"
  echo "$alive" | grep -q ALIVE && break || true
  sleep 5
done
lium exec brave-raven-a9 'bash -lc "echo ===; cat /root/affine_data/r781_train_launched.json 2>/dev/null; echo ===; ps -p $(cat /root/logs/r781_train.pid 2>/dev/null || echo 0) -o pid,etime,cmd 2>/dev/null | head -3; nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv; tail -3 /root/logs/r781_train.nohup"' 2>&1 | tee -a "$LOG"
log DONE
