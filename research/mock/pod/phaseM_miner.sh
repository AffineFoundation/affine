#!/usr/bin/env bash
# Track M MINER BOX bring-up (idempotent). B300 Blackwell -> Docker vLLM
# :latest (Track A's route; pip vLLM cannot JIT for sm_103a).
#   GPU 0  27B judge copy :8002 (runtime LoRA)   GPU 1  35B miner :8001
#   GPU 2,3  g_sft per round
set -uo pipefail
cd /root/work
W=/dshare/koth
mkdir -p "$W"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf

log() { echo "[phaseM $(date -u +%T)] $*"; }
note() { echo "$(date -u +%FT%TZ) [miner] $*" >> "$W/status.log"; }
healthy() { curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
wait_health() {
  local p="$1" t="${2:-1200}" i=0
  while ! healthy "$p"; do
    docker ps --format '{{.Names}}' | grep -q "vllm.*_$p\$" || return 1
    sleep 15; i=$((i+15)); [ "$i" -ge "$t" ] && return 1
  done
  log "port $p healthy after ${i}s"
}

# wait for model downloads
i=0
while ! grep -q SETUP_DONE /root/work/setup.log 2>/dev/null; do
  sleep 20; i=$((i+20))
  [ "$i" -ge 3600 ] && { log "setup not done after 60min; proceeding"; break; }
done
note "phaseM_miner: bringing up judge + miner servers"

if ! healthy 8002; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/koth bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.8-27B 8002 0 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-logprobs 25
fi
# second judge copy on GPU 4 (audit: GPUs 4-7 idle; halves reward-scoring time)
if ! healthy 8004; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/koth bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.8-27B 8004 4 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4 --max-logprobs 25
fi
if ! healthy 8001; then
  HF_DIR=/dshare/hf CKPT_DIR=/dshare/koth bash /root/work/pod_serve_docker.sh \
    Qwen/Qwen3.6-35B-A3B 8001 1 --max-model-len 32768 --enable-prefix-caching \
    --enable-lora --max-lora-rank 16 --max-loras 4
fi
wait_health 8002 1200 || { note "FATAL judge server :8002 failed"; exit 1; }
wait_health 8004 1200 || { note "FATAL judge2 server :8004 failed"; exit 1; }
wait_health 8001 1200 || { note "FATAL miner server :8001 failed"; exit 1; }
note "servers up: 0=judge :8002 | 4=judge2 :8004 | 1=miner :8001 | 2,3=g_sft (5-7 idle: downsize candidates)"

if ! pgrep -f 'miner_loop[.]py' >/dev/null; then
  nohup $PY /root/work/miner_loop.py > "$W/miner_driver.log" 2>&1 < /dev/null &
  disown
  note "miner loop launched (pid $!)"
fi
log "PHASEM_MINER_COMPLETE"
