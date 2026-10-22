#!/bin/bash
# usage: serve.sh CKPT_DIR PORT GPU_MEM_FRACTION [extra vllm args...]   (low CPU/IO priority; loopback only)
set -e
D=/root/affine-env-probe-20261022; CK=$1; PORT=$2; GMU=$3; shift 3
export VLLM_LOGGING_LEVEL=WARNING HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec nice -n 19 ionice -c3 $D/venv/bin/vllm serve "$CK" --host 127.0.0.1 --port "$PORT" --served-model-name probe \
  --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization "$GMU" --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 256 "$@"
