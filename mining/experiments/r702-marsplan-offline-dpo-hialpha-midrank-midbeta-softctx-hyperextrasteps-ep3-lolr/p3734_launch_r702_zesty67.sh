#!/usr/bin/env bash
# p3734: launch R702 TRAIN on idle zesty GPUs 6,7 (R691 REFUTE; R694 n80 on 4,5). Never pkill -f.
set -euo pipefail
EXP=r702-marsplan-offline-dpo-hialpha-midrank-midbeta-softctx-hyperextrasteps-ep3-lolr
log(){ echo "[p3734-r702-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r702_exp_p3734.tar.gz ]]; then
  echo "FATAL missing /tmp/r702_exp_p3734.tar.gz"; exit 1
fi

mkdir -p /root/mining_src
tar xzf /tmp/r702_exp_p3734.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_zesty_gpus67_p3734.sh

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 still busy used=$used"; exit 1; }

# Keep r694_merged (n80 live) + TKC — free oldest REFUTE if /tmp tight
df_avail=$(df -BG /tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
log "tmp_avail_G=$df_avail"
if [[ "${df_avail:-0}" -lt 200 ]]; then
  for d in /tmp/r686_merged /tmp/r691_merged /tmp/r662_merged /tmp/r674_merged /tmp/r668_merged; do
    if [[ -d "$d" ]]; then
      log "free $d"
      rm -rf "$d"
    fi
  done
fi

rm -f /root/logs/r702_lean_warm.log /root/logs/r702_lean_outer.nohup /root/logs/r702_wait_merge.nohup
nohup bash /root/mining_src/$EXP/lean_train_zesty_gpus67_p3734.sh >/root/logs/r702_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r702_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r702_train_then_merge_p3734.sh >/root/logs/r702_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r702_wait_merge.pid

for i in $(seq 1 90); do
  if [[ -f /root/logs/r702_train.pid ]]; then
    tpid=$(cat /root/logs/r702_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r702_lean_outer.pid) WAIT=$(cat /root/logs/r702_wait_merge.pid) TRAIN=$(cat /root/logs/r702_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r702_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r702_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
log DONE
