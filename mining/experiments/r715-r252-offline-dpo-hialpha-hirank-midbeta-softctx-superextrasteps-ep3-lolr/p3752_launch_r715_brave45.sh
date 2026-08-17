#!/usr/bin/env bash
set -euo pipefail
EXP=r715-r252-offline-dpo-hialpha-hirank-midbeta-softctx-superextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus45_p3752.sh >/root/logs/r715_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r715_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r715_train_then_merge_p3752.sh >/root/logs/r715_wait.nohup 2>&1 &
echo $! >/root/logs/r715_wait.pid
echo "launched lean=$(cat /root/logs/r715_lean_outer.pid) wait=$(cat /root/logs/r715_wait.pid)"
