#!/usr/bin/env bash
# p3725: launch R696 TRAIN on idle brave GPUs 4,5 (R688 MERGE_DONE). Never pkill -f.
set -euo pipefail
EXP=r696-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-ultraextrasteps-ep3-lolr
log(){ echo "[p3725-r696-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r696_exp_p3725.tar.gz ]]; then
  echo "FATAL missing /tmp/r696_exp_p3725.tar.gz"; exit 1
fi

mkdir -p /root/mining_src
tar xzf /tmp/r696_exp_p3725.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_brave_gpus45_p3725.sh

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  log "VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy used=$used"; exit 1; }

# Free oldest REFUTE merges if /tmp tight — keep r683+r688 (queued SCP sources)
df_avail=$(df -BG /tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
log "tmp_avail_G=$df_avail"
if [[ "${df_avail:-0}" -lt 200 ]]; then
  for d in /tmp/r679_merged /tmp/r681_merged /tmp/r637_merged; do
    if [[ -d "$d" ]]; then
      log "free $d"
      rm -rf "$d"
    fi
  done
fi

rm -f /root/logs/r696_lean_warm.log /root/logs/r696_lean_outer.nohup /root/logs/r696_wait_merge.nohup
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus45_p3725.sh >/root/logs/r696_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r696_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r696_train_then_merge_p3725.sh >/root/logs/r696_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r696_wait_merge.pid

for i in $(seq 1 90); do
  if [[ -f /root/logs/r696_train.pid ]]; then
    tpid=$(cat /root/logs/r696_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r696_lean_outer.pid) WAIT=$(cat /root/logs/r696_wait_merge.pid) TRAIN=$(cat /root/logs/r696_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r696_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 0,1,2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r696_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
log DONE
