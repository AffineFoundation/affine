#!/usr/bin/env bash
# p3786: if R739 MERGE_DONE and no n80 running, launch lean chall on zesty 6,7.
set -euo pipefail
MARK=/root/logs/r739_n80_launched.p3786
SCRIPT=/root/mining_src/r739-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-hyperextrasteps-ep3-lolr/lean_chall_n80_zesty_gpus67_p3786.sh
mkdir -p /root/logs
if [[ -f "$MARK" ]]; then echo "already launched"; exit 0; fi
if [[ ! -f /root/logs/r739_merge.done ]]; then echo "no merge.done"; exit 1; fi
if [[ ! -f /tmp/r739_merged/config.json ]]; then echo "no merge dir"; exit 1; fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$MARK"
nohup bash "$SCRIPT" >/root/logs/p3786_r739_lean.outer.log 2>&1 &
echo $! >/root/logs/p3786_r739_lean.outer.pid
echo "launched pid=$(cat /root/logs/p3786_r739_lean.outer.pid)"
