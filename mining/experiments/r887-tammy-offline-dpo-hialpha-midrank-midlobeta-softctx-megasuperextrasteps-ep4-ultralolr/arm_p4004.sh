#!/usr/bin/env bash
set -euo pipefail
if [[ -f /root/logs/p4004_r887_n80_armed.done ]]; then echo ALREADY_ARMED; exit 0; fi
used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
echo "GPU2_used=$used"
[[ "$used" -lt 2000 ]] || { echo FATAL_GPU2_BUSY; exit 1; }
SCRIPT=/root/mining_src/r887-tammy-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_brave_gpus2_tp1_p4004.sh
test -x "$SCRIPT"
nohup bash "$SCRIPT" >/root/logs/p4004_r887_chall.outer.nohup 2>&1 &
echo $! >/root/logs/p4004_r887_chall.outer.pid
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4004_r887_n80_armed.done
echo ARMED_PID=$(cat /root/logs/p4004_r887_chall.outer.pid)
sleep 10
tail -30 /root/logs/p4004_r887_chall_n80_wvk7.log 2>/dev/null || tail -30 /root/logs/p4004_r887_chall.outer.nohup
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv
ss -lntp | grep -E '8004|8002|8003' || true
