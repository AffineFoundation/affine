#!/usr/bin/env bash
# p3790: if R738 MERGE_DONE and no n80 running, launch lean chall on lunar 6,7.
set -euo pipefail
MARK=/root/logs/r738_n80_launched.p3790
SCRIPT=/root/mining_src/r738-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-superextrasteps-ep3-lolr/lean_chall_n80_lunar_gpus67_p3790.sh
mkdir -p /root/logs
if [[ -f "$MARK" ]]; then echo "already launched"; exit 0; fi
if [[ ! -f /root/logs/r738_merge.done ]]; then echo "no merge.done"; exit 1; fi
if [[ ! -f /tmp/r738_merged/config.json ]]; then echo "no merge dir"; exit 1; fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$MARK"
nohup bash "$SCRIPT" >/root/logs/p3790_r738_lean.outer.log 2>&1 &
echo $! >/root/logs/p3790_r738_lean.outer.pid
echo "launched pid=$(cat /root/logs/p3790_r738_lean.outer.pid)"
