#!/usr/bin/env bash
# p3904: fill idle brave 8×B200 with R821–R824 (keep /tmp/r809_merged+/tmp/r810_merged intact)
set -euo pipefail
ROOT=/home/const/subnet120/mining
E821=r821-tammy-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
E822=r822-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
E823=r823-tammy-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
E824=r824-tammy-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr
LOG=$ROOT/experiments/fleet-rent/logs/p3904_r821_r824_launch.log
TAR=/tmp/r821_r824_exp_p3904.tar.gz
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -p 40127 root@18.118.83.97)
SCP=(scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes -P 40127)
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3904-r821r824] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
chmod +x "$ROOT/experiments/$E821"/*.sh "$ROOT/experiments/$E822"/*.sh \
  "$ROOT/experiments/$E823"/*.sh "$ROOT/experiments/$E824"/*.sh
log "pack $E821 $E822 $E823 $E824"
tar czf "$TAR" -C "$ROOT/experiments" "$E821" "$E822" "$E823" "$E824"
ls -lh "$TAR" | tee -a "$LOG"
log "scp tar → brave"
"${SCP[@]}" "$TAR" root@18.118.83.97:/tmp/r821_r824_exp_p3904.tar.gz 2>&1 | tee -a "$LOG"
log "extract + launch ×4"
"${SSH[@]}" 'bash -s' <<REMOTE 2>&1 | tee -a "$LOG"
set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r821_r824_exp_p3904.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$E821/*.sh /root/mining_src/$E822/*.sh /root/mining_src/$E823/*.sh /root/mining_src/$E824/*.sh
# Do NOT touch /tmp/r809_merged or /tmp/r810_merged (active relay sources)
nohup bash /root/mining_src/$E823/lean_train_brave_gpus01_p3904.sh >/root/logs/p3904_r823_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3904_r823_lean.outer.pid
nohup bash /root/mining_src/$E823/wait_r823_train_then_merge_p3904.sh >/root/logs/p3904_r823_wait.nohup 2>&1 &
echo \$! >/root/logs/p3904_r823_wait.pid
nohup bash /root/mining_src/$E824/lean_train_brave_gpus23_p3904.sh >/root/logs/p3904_r824_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3904_r824_lean.outer.pid
nohup bash /root/mining_src/$E824/wait_r824_train_then_merge_p3904.sh >/root/logs/p3904_r824_wait.nohup 2>&1 &
echo \$! >/root/logs/p3904_r824_wait.pid
nohup bash /root/mining_src/$E821/lean_train_brave_gpus45_p3904.sh >/root/logs/p3904_r821_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3904_r821_lean.outer.pid
nohup bash /root/mining_src/$E821/wait_r821_train_then_merge_p3904.sh >/root/logs/p3904_r821_wait.nohup 2>&1 &
echo \$! >/root/logs/p3904_r821_wait.pid
nohup bash /root/mining_src/$E822/lean_train_brave_gpus67_p3904.sh >/root/logs/p3904_r822_lean.outer.nohup 2>&1 &
echo \$! >/root/logs/p3904_r822_lean.outer.pid
nohup bash /root/mining_src/$E822/wait_r822_train_then_merge_p3904.sh >/root/logs/p3904_r822_wait.nohup 2>&1 &
echo \$! >/root/logs/p3904_r822_wait.pid
echo LAUNCHED \
  r823=\$(cat /root/logs/p3904_r823_lean.outer.pid) \
  r824=\$(cat /root/logs/p3904_r824_lean.outer.pid) \
  r821=\$(cat /root/logs/p3904_r821_lean.outer.pid) \
  r822=\$(cat /root/logs/p3904_r822_lean.outer.pid)
REMOTE

for i in $(seq 1 60); do
  out=$("${SSH[@]}" 'bash -lc "
    for r in r821 r822 r823 r824; do
      echo -n \${r}_PID=; cat /root/logs/\${r}_train.pid 2>/dev/null || echo none
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    tail -1 /root/logs/r821_lean_warm.log 2>/dev/null || true
    tail -1 /root/logs/r822_lean_warm.log 2>/dev/null || true
    tail -1 /root/logs/r823_lean_warm.log 2>/dev/null || true
    tail -1 /root/logs/r824_lean_warm.log 2>/dev/null || true
  "' 2>&1 | tail -40)
  echo "poll=$i" | tee -a "$LOG"
  echo "$out" | tee -a "$LOG"
  ok=0
  echo "$out" | grep -q 'r821_PID=[0-9]' && echo "$out" | grep -q 'r822_PID=[0-9]' \
    && echo "$out" | grep -q 'r823_PID=[0-9]' && echo "$out" | grep -q 'r824_PID=[0-9]' && ok=1
  [[ "$ok" -eq 1 ]] && { log "all 4 TRAIN pids present"; break; }
  sleep 5
done
"${SSH[@]}" 'bash -lc "
  for r in r821 r822 r823 r824; do
    echo === \$r ===
    cat /root/affine_data/\${r}_train_launched.json 2>/dev/null | head -20 || true
    ps -p \$(cat /root/logs/\${r}_train.pid 2>/dev/null || echo 0) -o pid,etime,cmd 2>/dev/null | head -3 || true
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
"' 2>&1 | tee -a "$LOG"
log DONE
