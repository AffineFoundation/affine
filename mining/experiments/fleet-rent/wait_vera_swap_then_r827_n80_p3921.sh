#!/usr/bin/env bash
# p3921: after reign36 vera king-swap READY on crown, relaunch R827 n80 on
# idle GPUs 6,7 vs live vera6 (wvk=7). Prior R827 EngineDead vs tammy;
# merge kept at /tmp/r827_merged. Slot freed by R823 REFUTE (idle chall).
# Never pkill -f.
set -euo pipefail

LOG=/root/logs/wait_vera_swap_then_r827_n80_p3921.log
DONE=/root/logs/retarget_vera_reign36_p3918.done
CHALL=/root/mining_src/r827-tammy-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_reign36_p3921.sh
RELAUNCHED=/root/logs/r827_n80_relaunched.p3921
MERGE_DIR=/tmp/r827_merged

mkdir -p /root/logs
exec >>"$LOG" 2>&1

log(){ echo "[p3921-r827-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ -f "$RELAUNCHED" ]]; then
  log "already relaunched — exit"
  exit 0
fi

log "armed: wait vera swap DONE + :8001 serves vera6 → relaunch R827 n80 GPUs 6,7"

while true; do
  if [[ -f "$DONE" ]]; then
    kid=""
    if curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
      kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    fi
    log "done_stamp=1 king_id=${kid:-none}"
    if echo "${kid:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then
      n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ ! -f "$MERGE_DIR/config.json" ]] || [[ "${n:-0}" -lt 16 ]]; then
        log "FATAL merge missing shards=$n — abort"
        exit 2
      fi
      date -u +%Y-%m-%dT%H:%M:%SZ >"$RELAUNCHED"
      log "MERGE_OK shards=$n — relaunch chall+n80 vs reign36 vera"
      nohup bash "$CHALL" >/root/logs/p3921_r827_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3921_r827_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3921_r827_chall_n80.pid) log=/root/logs/p3921_r827_chall_n80_wvk7.log"
      exit 0
    fi
  else
    n=$(ls /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    inc=$(ls /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/blobs/*.incomplete 2>/dev/null | wc -l || true)
    log "waiting swap done_stamp=0 vera_shards=$n incomplete=$inc"
  fi
  sleep 20
done
