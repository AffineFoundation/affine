#!/usr/bin/env bash
set -euo pipefail
EXP=r704-r252-offline-dpo-hialpha-hirank-midbeta-softctx-hyperextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_brave_gpus67_p3736.sh >/root/logs/r704_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r704_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r704_train_then_merge_p3736.sh >/root/logs/r704_wait.nohup 2>&1 &
echo $! >/root/logs/r704_wait.pid
echo "launched lean=$(cat /root/logs/r704_lean_outer.pid) wait=$(cat /root/logs/r704_wait.pid)"
