#!/bin/bash
# Serve ONE model on all GPUs of the node as a single OpenAI-compatible endpoint (http://127.0.0.1:8000/v1, model name "bench"):
# vLLM data parallel, one engine replica per GPU (an 8-9B model fits one GPU; DP beats TP for throughput), internal load balancing.
# Any harness (vf-eval, lm-eval, curl) can target it. A model swap = stop this server (exact PID in serve.pid) and start it again.
# usage: serve.sh MODEL_DIR [REASONING_PARSER] [MAX_MODEL_LEN]
# Settings are part of the benchmark definition (bench/config/serving.toml) -- change them there and bump the version.
set -u
. /data/bench/env.sh
M=$1; RP=${2:-qwen3}; ML=${3:-34816}
# Same context for every model. GLM-Z1-9B-0414 is natively 32K (32768 positions); its 34816 window (2K prompt + 32K output)
# goes slightly past that. Disclosed in config/models.toml.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
W=/data/bench/serve; mkdir -p $W
NG=$(nvidia-smi -L | wc -l)
if [ -f $W/serve.pid ] && kill -0 "$(cat $W/serve.pid)" 2>/dev/null; then echo "already serving pid $(cat $W/serve.pid): stop first"; exit 2; fi
RPA=(); [ "$RP" != "none" ] && RPA=(--reasoning-parser "$RP")
TRC=(); grep -q '"auto_map"' "$M/config.json" 2>/dev/null && TRC=(--trust-remote-code)
echo "$M" > $W/model
setsid nohup /data/bench/venv-vllm/bin/vllm serve "$M" --host 127.0.0.1 --port 8000 --served-model-name bench \
  --data-parallel-size $NG --dtype bfloat16 --max-model-len $ML --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --max-num-seqs 512 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm "${RPA[@]}" "${TRC[@]}" > $W/serve.log 2>&1 < /dev/null &
echo $! > $W/serve.pid
for i in $(seq 720); do
  curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null && { date +%s > $W/serving_at; echo "serving $M pid $(cat $W/serve.pid)"; exit 0; }
  kill -0 "$(cat $W/serve.pid)" 2>/dev/null || { echo "server died"; tail -20 $W/serve.log; exit 4; }
  sleep 5
done
echo "timeout"; exit 5
