#!/usr/bin/env bash
set -euo pipefail
if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null; then echo king already up; exit 0; fi
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf CUDA_VISIBLE_DEVICES=5
KING=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
: >/root/logs/vllm_king.log
nohup /root/venv/bin/python3 /root/venv/bin/vllm serve "$KING" \
  --port 8001 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --max-model-len 65536 --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' --enforce-eager \
  --served-model-name vera6/affine-5g4yy75zuz-t6 \
  >/root/logs/vllm_king.log 2>&1 &
echo $! >/root/logs/vllm_king.pid
echo "king launched pid=$(cat /root/logs/vllm_king.pid)"
