#!/usr/bin/env bash
# Track S bring-up (idempotent). 8xB200 Blackwell -> Docker vLLM :latest
# (Track A/M's route; pip vLLM cannot JIT for sm_100).
#   GPU 0,1  27B TP2 :8000  sampler (teacher=base + G adapters, runtime LoRA)
#   GPU 2,3  27B TP2 :8001  judge   (base + D adapters, runtime LoRA)
#   GPU 4,5  trackS_d_train.py (DDP)    GPU 6,7  g_sft.py per round
set -uo pipefail
cd /root/work
W=/dshare/trackS
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf

log() { echo "[phaseS $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) [infra] $*" >> "$W/status.log"; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
wait_health() {
  local p="$1" t="${2:-1500}" i=0
  while ! healthy "$p"; do
    sleep 15; i=$((i+15)); [ "$i" -ge "$t" ] && return 1
  done
  log "port $p healthy after ${i}s"
}

# wait for model download
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 20; i=$((i+20))
  [ "$i" -ge 4200 ] && { log "setup not done after 70min; proceeding"; break; }
done

note "phaseS: bringing up sampler + judge servers (vllm :latest docker)"
if ! healthy 8000; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/trackS bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.8-27B 8000 0,1 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-logprobs 25
fi
if ! healthy 8001; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/trackS bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.8-27B 8001 2,3 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-logprobs 25
fi
wait_health 8000 1500 || { note "FATAL sampler :8000 failed"; exit 1; }
wait_health 8001 1500 || { note "FATAL judge :8001 failed"; exit 1; }
note "servers up: 0,1=sampler :8000 (teacher+G) | 2,3=judge :8001 (D) | 4,5=d_train | 6,7=g_sft"

if ! pgrep -f 'trackS_driver[.]py' >/dev/null; then
  nohup $PY /root/work/trackS_driver.py > "$W/driver.log" 2>&1 < /dev/null &
  disown
  note "trackS driver launched (pid $!)"
fi
log "PHASES_COMPLETE"
