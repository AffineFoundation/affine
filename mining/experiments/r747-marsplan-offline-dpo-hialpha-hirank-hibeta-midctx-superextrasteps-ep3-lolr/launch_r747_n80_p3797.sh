#!/usr/bin/env bash
# p3797: if R747 MERGE_DONE and no n80 running, launch lean chall on zesty 6,7.
set -euo pipefail
MARK=/root/logs/r747_n80_launched.p3797
SCRIPT=/root/mining_src/r747-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr/lean_chall_n80_zesty_gpus67_p3797.sh
mkdir -p /root/logs
if [[ -f "$MARK" ]]; then echo "already launched"; exit 0; fi
if [[ ! -f /root/logs/r747_merge.done ]]; then echo "no merge.done"; exit 1; fi
if [[ ! -f /tmp/r747_merged/config.json ]]; then echo "no merge dir"; exit 1; fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$MARK"
nohup bash "$SCRIPT" >/root/logs/p3797_r747_lean.outer.log 2>&1 &
echo $! >/root/logs/p3797_r747_lean.outer.pid
echo "launched pid=$(cat /root/logs/p3797_r747_lean.outer.pid)"
