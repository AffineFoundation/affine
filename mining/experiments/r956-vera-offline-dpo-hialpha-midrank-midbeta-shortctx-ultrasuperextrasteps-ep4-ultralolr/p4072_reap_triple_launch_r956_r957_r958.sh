#!/usr/bin/env bash
# p4072: crown triple REFUTE → exact-PID reap → R956/R957/R958 TRAIN
# Never pkill -f (matches SSH). Kill only listed chall PIDs + their children by PPID.
set -euo pipefail
exec >/root/logs/p4072_reap_r943_r945_r946_launch_r956_r957_r958.nohup 2>&1
echo "[p4072] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

kill_tree() {
  local pid="$1"
  [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[p4072] pid $pid already dead"; return 0
  fi
  local kids
  kids=$(pgrep -P "$pid" 2>/dev/null || true)
  for k in $kids; do kill_tree "$k"; done
  echo "[p4072] kill $pid"; kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4072] kill -9 $pid"; kill -9 "$pid" 2>/dev/null || true
  fi
}

for f in \
  /root/logs/vllm_chall_r943.pid \
  /root/logs/vllm_chall_r945.pid \
  /root/logs/vllm_chall_r946.pid \
  /root/logs/r943_sim_wvk7.pid \
  /root/logs/r945_sim_wvk7.pid \
  /root/logs/r946_sim_wvk7.pid
do
  if [[ -f "$f" ]]; then
    p=$(cat "$f" 2>/dev/null || true)
    echo "[p4072] reap from $f -> $p"
    kill_tree "$p"
  fi
done

# Hardcoded parents observed at p4072 (in case pidfiles stale)
for p in 86045 85930 86159; do
  kill_tree "$p"
done

echo "[p4072] wait VRAM free on 1,3/4,5/6,7"
for i in $(seq 1 120); do
  u13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4072] iter=$i u13=$u13 u45=$u45 u67=$u67"
  [[ "$u13" -lt 8192 && "$u45" -lt 8192 && "$u67" -lt 8192 ]] && break
  sleep 3
done
u13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
u45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
u67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$u13" -lt 8192 && "$u45" -lt 8192 && "$u67" -lt 8192 ]] || { echo "FATAL VRAM still busy"; nvidia-smi; exit 1; }

chmod +x \
  /root/mining_src/r956-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus45_p4072.sh \
  /root/mining_src/r956-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-ultralolr/wait_r956_train_then_merge_p4072.sh \
  /root/mining_src/r957-vera-offline-dpo-hialpha-lorank-hibeta-softctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus13_p4072.sh \
  /root/mining_src/r957-vera-offline-dpo-hialpha-lorank-hibeta-softctx-ultrasuperextrasteps-ep4-ultralolr/wait_r957_train_then_merge_p4072.sh \
  /root/mining_src/r958-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus67_p4072.sh \
  /root/mining_src/r958-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-ultralolr/wait_r958_train_then_merge_p4072.sh

nohup bash /root/mining_src/r957-vera-offline-dpo-hialpha-lorank-hibeta-softctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus13_p4072.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4072_r957_lean_outer.pid
nohup bash /root/mining_src/r956-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus45_p4072.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4072_r956_lean_outer.pid
nohup bash /root/mining_src/r958-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_crown_gpus67_p4072.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4072_r958_lean_outer.pid

sleep 8
echo "[p4072] outer pids r957=$(cat /root/logs/p4072_r957_lean_outer.pid) r956=$(cat /root/logs/p4072_r956_lean_outer.pid) r958=$(cat /root/logs/p4072_r958_lean_outer.pid)"
for id in r956 r957 r958; do
  echo "=== $id ==="
  tail -n 20 /root/logs/${id}_lean_warm.log 2>/dev/null || true
  [[ -f /root/logs/${id}_train.pid ]] && echo "train.pid=$(cat /root/logs/${id}_train.pid)" && ps -p "$(cat /root/logs/${id}_train.pid)" -o pid,cmd= || echo "train not yet"
done
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "[p4072] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
