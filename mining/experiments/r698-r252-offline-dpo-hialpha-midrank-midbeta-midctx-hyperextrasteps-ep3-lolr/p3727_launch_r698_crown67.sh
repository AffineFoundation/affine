#!/usr/bin/env bash
# p3727: launch R698 TRAIN on idle crown GPUs 6,7 (R692 MERGE_DONE). Never pkill -f.
set -euo pipefail
EXP=r698-r252-offline-dpo-hialpha-midrank-midbeta-midctx-hyperextrasteps-ep3-lolr
log(){ echo "[p3727-r698-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r698_exp_p3727.tar.gz ]]; then
  echo "FATAL missing /tmp/r698_exp_p3727.tar.gz"; exit 1
fi

mkdir -p /root/mining_src
tar xzf /tmp/r698_exp_p3727.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_crown_gpus67_p3727.sh

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy used=$used"; exit 1; }

# Keep r692 (just MERGE) + r693 train — free oldest REFUTE if /tmp tight
df_avail=$(df -BG /tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
log "tmp_avail_G=$df_avail"
if [[ "${df_avail:-0}" -lt 200 ]]; then
  for d in /tmp/r596_merged /tmp/r631_merged /tmp/r645_merged /tmp/r673_merged; do
    if [[ -d "$d" ]]; then
      log "free $d"
      rm -rf "$d"
    fi
  done
fi

rm -f /root/logs/r698_lean_warm.log /root/logs/r698_lean_outer.nohup /root/logs/r698_wait_merge.nohup
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus67_p3727.sh >/root/logs/r698_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r698_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r698_train_then_merge_p3727.sh >/root/logs/r698_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r698_wait_merge.pid

for i in $(seq 1 90); do
  if [[ -f /root/logs/r698_train.pid ]]; then
    tpid=$(cat /root/logs/r698_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r698_lean_outer.pid) WAIT=$(cat /root/logs/r698_wait_merge.pid) TRAIN=$(cat /root/logs/r698_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r698_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r698_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
log DONE
