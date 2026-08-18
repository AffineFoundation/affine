#!/usr/bin/env bash
# p3883: after R800 REFUTE, fill idle brave GPUs with R809 (4,5 MidCtx Loβ) + R810 (6,7 Soft Hi Lo UltraLoLR)
set -euo pipefail
ROOT=/home/const/subnet120/mining
E809=r809-tammy-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
E810=r810-tammy-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
LOG=$ROOT/experiments/fleet-rent/logs/p3883_r809_r810_launch.log
TAR=/tmp/r809_r810_exp_p3883.tar.gz
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -p 40127 root@18.118.83.97)
SCP=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes -P 40127)
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3883-r809r810] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
chmod +x "$ROOT/experiments/$E809"/*.sh "$ROOT/experiments/$E810"/*.sh
log "pack $E809 + $E810"
tar czf "$TAR" -C "$ROOT/experiments" "$E809" "$E810"
ls -lh "$TAR" | tee -a "$LOG"
log "scp tar → brave"
"${SCP[@]}" "$TAR" root@18.118.83.97:/tmp/r809_r810_exp_p3883.tar.gz 2>&1 | tee -a "$LOG"
log "extract + launch"
"${SSH[@]}" 'bash -s' <<REMOTE 2>&1 | tee -a "$LOG"
set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r809_r810_exp_p3883.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$E809/*.sh /root/mining_src/$E810/*.sh
# Keep /tmp/r801_merged intact (active host-relay source)
nohup bash /root/mining_src/$E809/lean_train_brave_gpus45_p3883.sh >/root/logs/p3883_r809_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3883_r809_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$E809/wait_r809_train_then_merge_p3883.sh >/root/logs/p3883_r809_wait.nohup 2>&1 &
echo \$! >/root/logs/p3883_r809_wait.pid
nohup bash /root/mining_src/$E810/lean_train_brave_gpus67_p3883.sh >/root/logs/p3883_r810_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3883_r810_lean.outer.pid
sleep 2
nohup bash /root/mining_src/$E810/wait_r810_train_then_merge_p3883.sh >/root/logs/p3883_r810_wait.nohup 2>&1 &
echo \$! >/root/logs/p3883_r810_wait.pid
echo LAUNCHED r809_lean=\$(cat /root/logs/p3883_r809_lean.outer.pid) r809_wait=\$(cat /root/logs/p3883_r809_wait.pid) r810_lean=\$(cat /root/logs/p3883_r810_lean.outer.pid) r810_wait=\$(cat /root/logs/p3883_r810_wait.pid)
REMOTE
for i in $(seq 1 48); do
  out=$("${SSH[@]}" 'bash -lc "
    echo -n R809_PID=; cat /root/logs/r809_train.pid 2>/dev/null || echo none
    echo -n R810_PID=; cat /root/logs/r810_train.pid 2>/dev/null || echo none
    tail -2 /root/logs/r809_lean_warm.log 2>/dev/null || true
    tail -2 /root/logs/r810_lean_warm.log 2>/dev/null || true
    nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  "' 2>&1 | tail -30)
  echo "poll=$i $out" | tee -a "$LOG"
  if echo "$out" | grep -q 'R809_PID=[0-9]' && echo "$out" | grep -q 'R810_PID=[0-9]'; then
    log "both TRAIN pids present"
    break
  fi
  sleep 5
done
"${SSH[@]}" 'bash -lc "
  echo === R809 ===; cat /root/affine_data/r809_train_launched.json 2>/dev/null | head -25
  echo === R810 ===; cat /root/affine_data/r810_train_launched.json 2>/dev/null | head -25
  ps -p \$(cat /root/logs/r809_train.pid 2>/dev/null || echo 0),\$(cat /root/logs/r810_train.pid 2>/dev/null || echo 0) -o pid,etime,cmd 2>/dev/null | head -5
  nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
"' 2>&1 | tee -a "$LOG"
log DONE
