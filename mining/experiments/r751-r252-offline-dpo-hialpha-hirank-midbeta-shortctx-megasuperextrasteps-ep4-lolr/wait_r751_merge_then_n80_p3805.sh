#!/usr/bin/env bash
# p3805: R751 MERGE already done (no prior wait→n80) → launch chall+n80 GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3805-r751-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r751_merge.done
MERGE_DIR=/tmp/r751_merged
LAUNCHED=/root/logs/r751_n80_launched.p3805
N80_SCRIPT=/root/mining_src/r751-r252-offline-dpo-hialpha-hirank-midbeta-shortctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_r252_gpus67_p3805.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]] || { log "FATAL merge not ready shards=$n"; exit 1; }
log "MERGE_DONE shards=$n — launch chall+n80 GPUs 6,7"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
nohup bash "$N80_SCRIPT" >/root/logs/p3805_r751_lean.outer.nohup 2>&1 &
echo $! >/root/logs/p3805_r751_lean.outer.pid
log "n80 outer pid=$(cat /root/logs/p3805_r751_lean.outer.pid)"
