#!/usr/bin/env bash
# Track P orchestration: start the three vLLM servers (with image fallback),
# then launch the driver. Idempotent: safe to re-run, skips healthy servers,
# driver resumes from state.json.
#
# GPU map (documented because `ps` does not show CUDA_VISIBLE_DEVICES):
#   GPU 0    G server   vllm  Qwen/Qwen3.6-35B-A3B  :8001  (+runtime LoRA)
#   GPU 1    T server   vllm  Qwen/Qwen3.8-27B      :8000
#   GPU 2,3  D server   vllm  Qwen/Qwen3.8-27B TP2  :8002  (+runtime LoRA)
#   GPU 4,5  D LoRA training   (subprocess per D round, driver-launched)
#   GPU 6,7  G SFT             (subprocess per round, driver-launched)
set -uo pipefail
cd /root/work
W=/dshare/gad
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf

log() { echo "[phaseP $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) [trackP] $*" >> "$W/status.log"; }

healthy() { curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }

wait_health() { # port timeout_s
  local p="$1" t="${2:-1200}" i=0
  while ! healthy "$p"; do
    # bail immediately if the container already died (bad image/arch)
    if ! docker ps --format '{{.Names}}' | grep -q "^vllm_$p\$"; then
      log "container vllm_$p exited"
      return 1
    fi
    sleep 15; i=$((i+15))
    if [ "$i" -ge "$t" ]; then return 1; fi
  done
  log "port $p healthy after ${i}s"
}

serve_fb() { # repo port gpus extra...
  local repo="$1" port="$2" gpus="$3"; shift 3
  if healthy "$port"; then log "port $port already healthy"; return 0; fi
  for img in v0.10.1.1 v0.11.0 latest; do
    local extra=()
    [ "$img" = "latest" ] && extra=(--enforce-eager --no-async-scheduling)
    log "serving $repo :$port gpus=$gpus image=$img"
    VLLM_IMAGE="vllm/vllm-openai:$img" \
      bash /root/work/serve_trackP.sh "$repo" "$port" "$gpus" "$@" "${extra[@]}"
    if wait_health "$port" 1200; then
      note "serving $repo :$port gpus=$gpus image=$img OK"
      return 0
    fi
    log "image $img failed for $repo; logs tail:"
    docker logs "vllm_$port" 2>&1 | tail -5
    docker rm -f "vllm_$port" >/dev/null 2>&1 || true
  done
  note "FATAL: could not serve $repo on :$port with any image"
  return 1
}

note "phaseP starting: bringing up G/T/D servers"

serve_fb Qwen/Qwen3.6-35B-A3B 8001 0 \
  --max-model-len 32768 --enable-lora --max-lora-rank 16 --max-loras 4 &
PG=$!
serve_fb Qwen/Qwen3.8-27B 8000 1 --max-model-len 32768 &
PT=$!
serve_fb Qwen/Qwen3.8-27B 8002 2,3 \
  --max-model-len 8192 --enable-lora --max-lora-rank 16 --max-loras 4 \
  --max-logprobs 25 &
PD=$!
wait $PG || exit 1
wait $PD || exit 1
wait $PT || true   # teacher server is non-critical (rollouts are cached)

note "servers up; launching driver"
if ! pgrep -f 'trackP_driver[.]py' >/dev/null; then
  nohup $PY /root/work/trackP_driver.py > "$W/driver.log" 2>&1 < /dev/null &
  disown
  note "driver launched (pid $!)"
else
  note "driver already running"
fi
log "PHASEP_COMPLETE"
