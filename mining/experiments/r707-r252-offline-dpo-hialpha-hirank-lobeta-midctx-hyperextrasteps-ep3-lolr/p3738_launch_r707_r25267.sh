#!/usr/bin/env bash
set -euo pipefail
EXP=r707-r252-offline-dpo-hialpha-hirank-lobeta-midctx-hyperextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3738.sh >/root/logs/r707_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r707_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r707_train_then_merge_p3738.sh >/root/logs/r707_wait.nohup 2>&1 &
echo $! >/root/logs/r707_wait.pid
echo "launched lean=$(cat /root/logs/r707_lean_outer.pid) wait=$(cat /root/logs/r707_wait.pid)"
