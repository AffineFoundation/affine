#!/usr/bin/env bash
set -euo pipefail
EXP=r705-r252-offline-dpo-hialpha-hirank-hibeta-softctx-hyperextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_crown_gpus45_p3737.sh >/root/logs/r705_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r705_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r705_train_then_merge_p3737.sh >/root/logs/r705_wait.nohup 2>&1 &
echo $! >/root/logs/r705_wait.pid
echo "launched lean=$(cat /root/logs/r705_lean_outer.pid) wait=$(cat /root/logs/r705_wait.pid)"
