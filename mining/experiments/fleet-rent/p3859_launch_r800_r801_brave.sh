#!/usr/bin/env bash
# p3859 host→brave: tar-upload R800+R801 + launch trains on GPUs 4,5 + 6,7
set -euo pipefail
ROOT=/home/const/subnet120/mining
E800=r800-tammy-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
E801=r801-tammy-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
LOG=$ROOT/experiments/fleet-rent/logs/p3859_r800_r801_launch.log
TAR=/tmp/r800_r801_exp_p3859.tar.gz
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -p 40127 root@18.118.83.97)
SCP=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes -P 40127)
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3859-r800r801] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
chmod +x "$ROOT/experiments/$E800"/*.sh "$ROOT/experiments/$E801"/*.sh
log "pack $E800 + $E801"
tar czf "$TAR" -C "$ROOT/experiments" "$E800" "$E801"
ls -lh "$TAR" | tee -a "$LOG"
log "scp tar → brave"
"${SCP[@]}" "$TAR" root@18.118.83.97:/tmp/r800_r801_exp_p3859.tar.gz 2>&1 | tee -a "$LOG"
log "extract + launch"
"${SSH[@]}" 'bash -s' <<REMOTE 2>&1 | tee -a "$LOG"
set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r800_r801_exp_p3859.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$E800/*.sh /root/mining_src/$E801/*.sh
# R800 GPUs 4,5
nohup bash /root/mining_src/$E800/lean_train_brave_gpus45_p3859.sh >/root/logs/p3859_r800_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3859_r800_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$E800/wait_r800_train_then_merge_p3859.sh >/root/logs/p3859_r800_wait.nohup 2>&1 &
echo \$! >/root/logs/p3859_r800_wait.pid
# R801 GPUs 6,7
nohup bash /root/mining_src/$E801/lean_train_brave_gpus67_p3859.sh >/root/logs/p3859_r801_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3859_r801_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$E801/wait_r801_train_then_merge_p3859.sh >/root/logs/p3859_r801_wait.nohup 2>&1 &
echo \$! >/root/logs/p3859_r801_wait.pid
echo LAUNCHED r800_lean=\$(cat /root/logs/p3859_r800_lean.outer.pid) r800_wait=\$(cat /root/logs/p3859_r800_wait.pid) r801_lean=\$(cat /root/logs/p3859_r801_lean.outer.pid) r801_wait=\$(cat /root/logs/p3859_r801_wait.pid)
REMOTE
for i in $(seq 1 36); do
  out=$("${SSH[@]}" 'bash -lc "
    echo -n R800_PID=; cat /root/logs/r800_train.pid 2>/dev/null || echo none
    echo -n R801_PID=; cat /root/logs/r801_train.pid 2>/dev/null || echo none
    tail -3 /root/logs/r800_lean_warm.log 2>/dev/null || true
    tail -3 /root/logs/r801_lean_warm.log 2>/dev/null || true
    nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  "' 2>&1 | tail -25)
  echo "poll=$i $out" | tee -a "$LOG"
  if echo "$out" | grep -q 'R800_PID=[0-9]' && echo "$out" | grep -q 'R801_PID=[0-9]'; then
    log "both TRAIN pids present"
    break
  fi
  sleep 5
done
"${SSH[@]}" 'bash -lc "
  echo === R800 ===; cat /root/affine_data/r800_train_launched.json 2>/dev/null | head -20
  echo === R801 ===; cat /root/affine_data/r801_train_launched.json 2>/dev/null | head -20
  ps -p \$(cat /root/logs/r800_train.pid 2>/dev/null || echo 0),\$(cat /root/logs/r801_train.pid 2>/dev/null || echo 0) -o pid,etime,cmd 2>/dev/null | head -5
  nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
"' 2>&1 | tee -a "$LOG"
log DONE
