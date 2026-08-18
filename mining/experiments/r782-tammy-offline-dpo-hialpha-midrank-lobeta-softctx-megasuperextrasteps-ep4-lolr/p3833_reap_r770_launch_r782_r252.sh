#!/usr/bin/env bash
# p3833: R770 REFUTE → reap chall by pid → launch R782 TRAIN 6,7 + wait→merge + wait→n80.
# Keep R761 wait on 4,5. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=r782-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-lolr
SRC=$ROOT/experiments/$EXP
LOG=$ROOT/experiments/fleet-rent/logs/p3833_r770_reap_r782_launch.log
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3833-reap-r770] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
source /home/const/subnet120/.venv/bin/activate
log "pack $EXP"
TAR=/tmp/r782_exp_p3833.tar.gz
tar czf "$TAR" -C "$ROOT/experiments" "$EXP"
ls -lh "$TAR" | tee -a "$LOG"
log "scp → R252"
lium scp gentle-wolf-8c "$TAR" /tmp/r782_exp_p3833.tar.gz 2>&1 | tee -a "$LOG"
log "extract + reap R770 + launch R782"
lium exec gentle-wolf-8c "bash -lc 'set -euo pipefail
log(){ echo \"[p3833-reap-r770] \$(date -u +%Y-%m-%dT%H:%M:%SZ) \$*\"; }
stop_pid() {
  local pid=\$1; local why=\${2:-}
  [[ -n \"\${pid:-}\" && \"\$pid\" =~ ^[0-9]+\$ ]] || return 0
  if kill -0 \"\$pid\" 2>/dev/null; then
    log \"kill pid=\$pid (\$why)\"
    kill \"\$pid\" 2>/dev/null || true
    for _ in \$(seq 1 40); do kill -0 \"\$pid\" 2>/dev/null || break; sleep 1; done
    kill -9 \"\$pid\" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=\$1; local why=\${2:-}
  [[ -f \"\$pidf\" ]] || return 0
  local pid; pid=\$(cat \"\$pidf\" 2>/dev/null || true)
  stop_pid \"\$pid\" \"\$why pidf=\$pidf\"
  rm -f \"\$pidf\"
}
mkdir -p /root/mining_src /root/logs
tar xzf /tmp/r782_exp_p3833.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
log \"START reap R770 chall; keep /tmp/r770_merged; leave R761 wait 4,5\"
stop_pidfile /root/logs/r770_sim_wvk7.pid \"r770 sim\"
stop_pidfile /root/logs/p3819_r770_lean.outer.pid \"r770 lean outer\"
stop_pidfile /root/logs/p3819_r770_lean_outer.pid \"r770 lean outer2\"
stop_pidfile /root/logs/vllm_chall_r770.pid \"r770 vllm\"
stop_pidfile /root/logs/p3819_r770_chall_n80.pid \"r770 chall\"
while read -r pid; do
  [[ \"\$pid\" =~ ^[0-9]+\$ ]] || continue
  stop_pid \"\$pid\" \"r770 argv\"
done < <(ps -eo pid=,args= | awk \"/vllm serve .*\\/tmp\\/r770_merged|run_sim_duel.py.*r770|lean_chall_n80_r252_gpus67_p3819/ && !/awk/ {print \\\$1}\")
for i in \$(seq 1 90); do
  used67=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
  log \"wait free used67=\$used67 iter=\$i\"
  [[ \"\${used67:-999999}\" -lt 8192 ]] && break
  sleep 2
done
used67=\$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk \"{s+=\\\$1} END{print s+0}\")
[[ \"\${used67:-999999}\" -lt 8192 ]] || { log \"FATAL GPUs 6,7 still busy\"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r770_refute_reaped.p3833
log \"launch R782 TRAIN 6,7 + wait→merge + wait→n80\"
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3833.sh >/root/logs/p3833_r782_lean_outer.nohup 2>&1 &
echo \$! >/root/logs/p3833_r782_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r782_train_then_merge_p3833.sh >/root/logs/p3833_r782_wait.nohup 2>&1 &
echo \$! >/root/logs/p3833_r782_wait.pid
nohup bash /root/mining_src/$EXP/wait_r782_merge_then_n80_p3833.sh >/root/logs/p3833_r782_wait_n80.nohup 2>&1 &
echo \$! >/root/logs/p3833_r782_wait_n80.pid
log \"R782 lean=\$(cat /root/logs/p3833_r782_lean_outer.pid) wait=\$(cat /root/logs/p3833_r782_wait.pid) wait_n80=\$(cat /root/logs/p3833_r782_wait_n80.pid)\"
echo LEAN=\$(cat /root/logs/p3833_r782_lean_outer.pid)
'" 2>&1 | tee -a "$LOG"

for i in $(seq 1 36); do
  out=$(lium exec gentle-wolf-8c 'bash -lc "test -f /root/logs/r782_train.pid && cat /root/logs/r782_train.pid; tail -5 /root/logs/r782_lean_warm.log 2>/dev/null; nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader"' 2>&1 | tail -20)
  echo "poll=$i $out" | tee -a "$LOG"
  alive=$(lium exec gentle-wolf-8c 'bash -lc "pid=$(cat /root/logs/r782_train.pid 2>/dev/null || echo); [[ -n $pid ]] && kill -0 $pid 2>/dev/null && echo ALIVE:$pid || echo DEAD"' 2>&1 | tail -3)
  echo "alive=$alive" | tee -a "$LOG"
  echo "$alive" | grep -q ALIVE && break || true
  sleep 5
done
lium exec gentle-wolf-8c 'bash -lc "cat /root/affine_data/r782_train_launched.json 2>/dev/null; nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv; test -f /root/logs/r770_refute_reaped.p3833 && echo R770_REAPED; ps -p $(cat /root/logs/p3830_r761_wait.pid 2>/dev/null || echo 0) -o pid=,cmd= 2>/dev/null | head -1"' 2>&1 | tee -a "$LOG"
log DONE
