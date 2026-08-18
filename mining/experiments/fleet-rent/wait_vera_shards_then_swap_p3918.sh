#!/usr/bin/env bash
# Wait for vera6 2-shard snapshot, then kill stale retarget and swap king with fixed hub_ok.
set -euo pipefail
LOG=/root/logs/wait_vera_shards_then_swap_p3918.log
exec >>"$LOG" 2>&1
SNAP=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
log(){ echo "[p3918-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "armed wait 2 shards at $SNAP"
while true; do
  n=$(ls "$SNAP"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  inc=$(ls /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/blobs/*.incomplete 2>/dev/null | wc -l || true)
  # du can fail before blobs/ exists (pipefail would kill the waiter) — tolerate
  sz=$(du -sb /root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/blobs 2>/dev/null | awk '{print $1}' || true)
  sz=${sz:-0}
  log "shards=$n incomplete=$inc blobs_bytes=$sz"
  if [[ "${n:-0}" -ge 2 ]]; then
    log "SIZE_OK — stop stale retarget outer if still on prefetch"
    # kill exact outer pid if alive (prefetch may still be in hub_ok≥16 fail path)
    if [[ -f /root/logs/retarget_vera_reign36_p3918.outer.pid ]]; then
      op=$(cat /root/logs/retarget_vera_reign36_p3918.outer.pid)
      if [[ "$op" =~ ^[0-9]+$ ]] && kill -0 "$op" 2>/dev/null; then
        # only kill if not already past swap (done file absent)
        if [[ ! -f /root/logs/retarget_vera_reign36_p3918.done ]]; then
          log "kill outer $op for relaunch with hub_ok≥2"
          kill "$op" 2>/dev/null || true
          sleep 2
          kill -9 "$op" 2>/dev/null || true
        fi
      fi
    fi
    # also kill child bash retarget if hung in hub_ok loop — by exact cmdline
    while read -r pid; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -q 'retarget_king_vera_reign36_p3918.sh'; then
        log "kill retarget bash pid=$pid"
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
      fi
    done < <(ps -eo pid=,args= | awk '/retarget_king_vera_reign36_p3918/ && !/wait_vera|awk/ {print $1}')
    set -a; source /root/.hf_token_p3918.env 2>/dev/null || true; set +a
    export PREFETCH=0 SWAP_KING=1
    log "launch fixed swap PREFETCH=0"
    nohup bash /root/mining_src/fleet-rent/retarget_king_vera_reign36_p3918.sh \
      >/root/logs/retarget_vera_reign36_p3918.swap.nohup 2>&1 &
    echo $! > /root/logs/retarget_vera_reign36_p3918.swap.pid
    log "swap pid=$(cat /root/logs/retarget_vera_reign36_p3918.swap.pid)"
    exit 0
  fi
  sleep 30
done
