#!/bin/bash
# usage: serve.sh CKPT_DIR PORT GPU_MEM_FRACTION [extra vllm args...]   (low CPU/IO priority; loopback only)
set -e
D=/root/affine-env-probe; CK=$1; PORT=$2; GMU=$3; shift 3
# replicas >0 wait for replica 0 (shared torch.compile/flashinfer caches race when 8 compile at once)
if [ -n "$WAIT_PORT" ]; then for _ in $(seq 240); do curl -sf -m 3 http://127.0.0.1:$WAIT_PORT/v1/models >/dev/null && break; sleep 5; done; fi
# 2026-10-29 (red-skip worker): per-replica compile caches. The shared ~/.cache/vllm inductor/triton dirs raced at RUNTIME
# (lazy top-p kernel compile, e243: FileNotFoundError .../inductor_cache/triton/0/<hash>/_topp_s* -> EngineDeadError on 2 replicas).
# Each replica (keyed by its GPU) gets its own VLLM_CACHE_ROOT + TRITON_CACHE_DIR + TORCHINDUCTOR_CACHE_DIR, hard-link seeded once
# from the shared cache (cache writers use tmp+rename, so a replica never changes another's files). Revert: serve.sh.bak-pre-percache-*.
R=${CUDA_VISIBLE_DEVICES:-x}; C=$D/vcache/r$R
if [ ! -e "$C/.seeded" ]; then mkdir -p "$C"; [ -d /root/.cache/vllm ] && nice -n 19 cp -al /root/.cache/vllm/. "$C/vllm/" 2>/dev/null || true; touch "$C/.seeded"; fi
mkdir -p "$C/vllm" "$C/triton" "$C/inductor"
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor
export VLLM_LOGGING_LEVEL=WARNING HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec $D/venv/bin/vllm serve "$CK" --host 127.0.0.1 --port "$PORT" --served-model-name probe \
  --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization "$GMU" --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs ${MAX_SEQS:-512} "$@"
