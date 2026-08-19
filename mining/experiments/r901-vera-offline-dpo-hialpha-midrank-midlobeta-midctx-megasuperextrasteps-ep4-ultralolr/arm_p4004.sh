#!/usr/bin/env bash
set -euo pipefail
EXP=/root/mining_src/r901-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r901 /root/logs
test -s /root/r901/dpo_duel_reason.jsonl || cp -f "$EXP/dpo_duel_reason.jsonl" /root/r901/dpo_duel_reason.jsonl
if [[ -f /root/logs/p4004_r901_armed.done ]]; then
  echo ALREADY_ARMED
  cat /root/logs/r901_train.pid 2>/dev/null || true
  exit 0
fi
bash "$EXP/lean_train_crown_gpus67_p4004.sh"
nohup bash "$EXP/wait_r901_train_then_merge_p4004.sh" >/root/logs/p4004_r901_wait.nohup 2>&1 &
echo $! >/root/logs/p4004_r901_wait.pid
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4004_r901_armed.done
echo TRAIN_PID=$(cat /root/logs/r901_train.pid)
echo WAIT_PID=$(cat /root/logs/p4004_r901_wait.pid)
tail -20 /root/logs/r901_lean_warm.log
sleep 2
tail -10 /root/logs/r901_train.nohup || true
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
