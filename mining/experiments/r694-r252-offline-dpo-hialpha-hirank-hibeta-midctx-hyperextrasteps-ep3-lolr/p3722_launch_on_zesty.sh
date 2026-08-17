#!/usr/bin/env bash
# p3722: unpack+launch R694 on zesty GPUs 4,5 (after R686 REFUTE reap)
set -euo pipefail
EXP=r694-r252-offline-dpo-hialpha-hirank-hibeta-midctx-hyperextrasteps-ep3-lolr
log(){ echo "[p3722-r694-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r694_exp_p3722.tar.gz ]]; then
  echo "FATAL missing /tmp/r694_exp_p3722.tar.gz"; exit 1
fi
mkdir -p /root/mining_src
tar xzf /tmp/r694_exp_p3722.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_zesty_gpus45_p3722.sh

# ensure GPUs 4,5 free (chall should already be reaped)
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  log "VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy used=$used"; exit 1; }

rm -f /root/logs/r694_lean_warm.log /root/logs/r694_lean_outer.nohup /root/logs/r694_wait_merge.nohup
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus45_p3722.sh >/root/logs/r694_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r694_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r694_train_then_merge_p3722.sh >/root/logs/r694_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r694_wait_merge.pid

for i in $(seq 1 45); do
  if [[ -f /root/logs/r694_train.pid ]]; then
    tpid=$(cat /root/logs/r694_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r694_lean_outer.pid) WAIT=$(cat /root/logs/r694_wait_merge.pid) TRAIN=$(cat /root/logs/r694_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r694_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 4,5 --query-gpu=memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r694_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
ps -p 805534 -o pid,etime,cmd 2>/dev/null | head -2 || true
ps aux | grep "tar cf - r680" | grep -v grep | head -2 || true
log DONE
