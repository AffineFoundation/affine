#!/usr/bin/env bash
# Every-5-rounds SWE bench on the newest eligible G checkpoint (GPU 3).
# Serialized: one merge+serve+bench at a time. Skips checkpoints already run.
set -uo pipefail
W=/dshare/gad
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export PATH=/root/benchenv/bin:$PATH

note() { echo "$(date -u +%FT%TZ) $*" >> "$W/status.log"; }
log() { echo "[swe_watch $(date -u +%T)] $*"; }

wait_health() {
  local p="$1" t="${2:-900}" i=0
  while ! curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; do
    sleep 10; i=$((i+10)); [ "$i" -ge "$t" ] && return 1
  done
  return 0
}

while true; do
  R=$($PY -c "import json;print(json.load(open('$W/state.json')).get('round',0))" 2>/dev/null || echo 0)
  # newest completed multiple-of-5 checkpoint not yet benched
  K=""
  for c in 60 55 50 45 40 35 30 25 20 15 10 5; do
    if [ "$c" -lt "$R" ] && [ -d "$W/g_lora/r$c" ] && [ ! -s "$W/swe/swe_r$c.json" ]; then
      K=$c; break
    fi
  done
  if [ -z "$K" ]; then sleep 300; continue; fi

  log "benching checkpoint r$K (driver round=$R)"
  MERGED="$W/swe/merged_r$K"
  if [ ! -d "$MERGED" ]; then
    CUDA_VISIBLE_DEVICES=3 $PY /root/work/merge_lora.py \
      --lora "$W/g_lora/r$K" --out "$MERGED" >> "$W/swe_watcher.log" 2>&1 || { sleep 120; continue; }
  fi
  bash /root/work/serve_h200.sh "$MERGED" 8030 3 --enforce-eager
  if wait_health 8030 900; then
    /root/benchenv/bin/python /root/work/run_swe.py \
      --model "$MERGED" --port 8030 --workers 24 --tag "r$K" \
      > "$W/swe_r${K}_run.log" 2>&1
    note "[trackB] SWE r$K: $(cat "$W/swe/swe_r$K.json" 2>/dev/null | head -c 300)"
  else
    note "[trackB] SWE r$K: server failed to become healthy"
  fi
  docker rm -f vllm_8030 >/dev/null 2>&1 || true
  sleep 20
done
