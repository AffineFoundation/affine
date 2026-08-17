#!/usr/bin/env bash
set -euo pipefail
EXP=r716-r252-offline-dpo-hialpha-hirank-hibeta-softctx-superextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus67_p3752.sh >/root/logs/r716_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r716_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r716_train_then_merge_p3752.sh >/root/logs/r716_wait.nohup 2>&1 &
echo $! >/root/logs/r716_wait.pid
echo "launched lean=$(cat /root/logs/r716_lean_outer.pid) wait=$(cat /root/logs/r716_wait.pid)"
