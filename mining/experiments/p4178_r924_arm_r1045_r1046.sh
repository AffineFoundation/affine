#!/usr/bin/env bash
# p4178: arm R1045 (GPUs6,7 SoftCtx Ultra) + R1046 (GPUs4,5 MidCtx Mega) on mine-r924. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4178_r924_arm_r1045_r1046.log
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4178] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"

# TK must stay warm
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4178] TK warm"

for rid in r1045 r1046; do
  EXPDIR=$(ls -d /root/mining_src/${rid}-vera-offline-dpo-* 2>/dev/null | head -1)
  test -n "$EXPDIR"
  chmod +x "$EXPDIR"/*.sh
done

nohup bash /root/mining_src/r1046-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-midlr/lean_train_r924_gpus45_p4178.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4178_r1046_launch.pid
sleep 2
nohup bash /root/mining_src/r1045-vera-offline-dpo-hialpha-hirank-hibeta-softctx-ultrasuperextrasteps-ep4-midlr/lean_train_r924_gpus67_p4178.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4178_r1045_launch.pid

sleep 8
for rid in r1045 r1046; do
  echo "=== $rid warm ==="; tail -40 /root/logs/${rid}_lean_warm.log || true
  echo "=== $rid train.pid ==="; cat /root/logs/${rid}_train.pid 2>/dev/null || true
  echo "=== $rid train head ==="; head -20 /root/logs/${rid}_train.nohup 2>/dev/null || true
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4178_r924_r1045_r1046_armed.done
echo "[p4178] DONE armed"
