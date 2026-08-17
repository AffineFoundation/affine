#!/usr/bin/env bash
# p3724: reap idle R596 chall (:8002 / GPUs 4,5) by pidfile, then launch R695 TRAIN.
# Never pkill -f. Keep /tmp/r596_merged for reference.
set -euo pipefail
EXP=r695-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-ultraextrasteps-ep3-lolr
log(){ echo "[p3724-r695-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ ! -f /tmp/r695_exp_p3724.tar.gz ]]; then
  echo "FATAL missing /tmp/r695_exp_p3724.tar.gz"; exit 1
fi

# Reap R596 chall only if n80 done and still listening on :8002
if [[ -f /root/logs/r596_reign34_wvk7_n80.done ]]; then
  vpid=""
  if [[ -f /root/logs/vllm_chall_r596.pid ]]; then
    vpid=$(cat /root/logs/vllm_chall_r596.pid 2>/dev/null || true)
  fi
  if [[ -z "${vpid:-}" ]] || ! [[ "$vpid" =~ ^[0-9]+$ ]]; then
    # fallback: find serve on :8002 for r596_merged
    vpid=$(ss -lntp 2>/dev/null | awk '/:8002/ {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)
  fi
  if [[ -n "${vpid:-}" && "$vpid" =~ ^[0-9]+$ ]] && kill -0 "$vpid" 2>/dev/null; then
    cmd=$(ps -p "$vpid" -o args= 2>/dev/null || true)
    if echo "$cmd" | grep -q 'r596_merged\|:8002\|8002'; then
      log "reap R596 chall pid=$vpid"
      kill "$vpid" 2>/dev/null || true
      for i in $(seq 1 60); do
        kill -0 "$vpid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$vpid" 2>/dev/null; then
        log "escalate TERM→KILL pid=$vpid"
        kill -9 "$vpid" 2>/dev/null || true
      fi
      date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r596_chall_reaped_p3724.done
      log "R596 chall reaped"
    else
      log "skip kill pid=$vpid cmd_mismatch"
    fi
  else
    log "no live R596 chall pid"
  fi
else
  echo "FATAL R596 wvk7 n80 not done — refuse reap"; exit 1
fi

mkdir -p /root/mining_src
tar xzf /tmp/r695_exp_p3724.tar.gz -C /root/mining_src
chmod +x /root/mining_src/$EXP/*.sh
test -f /root/mining_src/$EXP/lean_train_r252_gpus45_p3724.sh

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  log "VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 still busy used=$used"; exit 1; }

# Do not free /tmp/r596_merged (kept). Free older REFUTE merges if /tmp tight — skip for now.

rm -f /root/logs/r695_lean_warm.log /root/logs/r695_lean_outer.nohup /root/logs/r695_wait_merge.nohup
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3724.sh >/root/logs/r695_lean_outer.nohup 2>&1 &
echo $! > /root/logs/r695_lean_outer.pid
sleep 3
nohup bash /root/mining_src/$EXP/wait_r695_train_then_merge_p3724.sh >/root/logs/r695_wait_merge.nohup 2>&1 &
echo $! > /root/logs/r695_wait_merge.pid

for i in $(seq 1 60); do
  if [[ -f /root/logs/r695_train.pid ]]; then
    tpid=$(cat /root/logs/r695_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_UP pid=$tpid iter=$i"
      break
    fi
  fi
  sleep 2
done

echo "OUTER=$(cat /root/logs/r695_lean_outer.pid) WAIT=$(cat /root/logs/r695_wait_merge.pid) TRAIN=$(cat /root/logs/r695_train.pid 2>/dev/null || echo none)"
echo "=== lean_warm ==="
cat /root/logs/r695_lean_warm.log || true
echo "=== gpu ==="
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
ps -p "$(cat /root/logs/r695_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null | head -3 || true
ps -p 268426 -o pid,etime,cmd 2>/dev/null | head -2 || true
ps aux | grep "tar cf - r675" | grep -v grep | head -2 || true
ss -lntp | grep -E ':800[0-9]' || true
log DONE
