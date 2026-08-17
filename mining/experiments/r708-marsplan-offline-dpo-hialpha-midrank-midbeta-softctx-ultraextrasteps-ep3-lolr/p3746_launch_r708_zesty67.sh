#!/usr/bin/env bash
# p3746: launch R708 TRAIN on idle zesty GPUs 6,7 (R702 REFUTE; R703 n80 on 4,5). Never pkill -f.
set -euo pipefail
EXP=r708-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-ultraextrasteps-ep3-lolr
log(){ echo "[p3746-r708-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r708_exp_p3746.tar.gz ]]; then
  echo "FATAL missing /tmp/r708_exp_p3746.tar.gz"; exit 1
fi

mkdir -p /root/mining_src
tar xzf /tmp/r708_exp_p3746.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_zesty_gpus67_p3746.sh

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy used=$used"; exit 1; }

# Keep r703_merged (n80 live) + r702 + TKC — free oldest REFUTE if /tmp tight
df_avail=$(df -BG /tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
log "tmp_avail_G=$df_avail"
if [[ "${df_avail:-0}" -lt 200 ]]; then
  for d in /tmp/r668_merged /tmp/r680_merged; do
    if [[ -d "$d" ]]; then
      log "free $d"
      rm -rf "$d"
    fi
  done
fi

rm -f /root/logs/r708_lean_warm.log /root/logs/r708_lean_outer.nohup /root/logs/r708_wait_merge.nohup
rm -f /root/logs/r708_merge_launched.p3746
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3746.sh >/root/logs/r708_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r708_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r708_train_then_merge_p3746.sh >/root/logs/r708_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r708_wait_merge.pid

for i in $(seq 1 90); do
  if [[ -f /root/logs/r708_train.pid ]]; then
    tpid=$(cat /root/logs/r708_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r708_lean_outer.pid) WAIT=$(cat /root/logs/r708_wait_merge.pid) TRAIN=$(cat /root/logs/r708_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r708_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r708_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
log DONE
