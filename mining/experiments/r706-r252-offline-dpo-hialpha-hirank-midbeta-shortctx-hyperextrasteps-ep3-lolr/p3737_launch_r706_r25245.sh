#!/usr/bin/env bash
set -euo pipefail
EXP=r706-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-hyperextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus45_p3737.sh >/root/logs/r706_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r706_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r706_train_then_merge_p3737.sh >/root/logs/r706_wait.nohup 2>&1 &
echo $! >/root/logs/r706_wait.pid
echo "launched lean=$(cat /root/logs/r706_lean_outer.pid) wait=$(cat /root/logs/r706_wait.pid)"
