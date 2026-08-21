#!/usr/bin/env bash
# p4227: relaunch king on GPU5 — p4226 start died (flashinfer JIT: no CUDA_HOME/nvcc).
set -euo pipefail
if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null; then
  echo "king already up"
  exit 0
fi
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
_CU13="${_SITE}/nvidia/cu13"
if [[ -x "${_CU13}/bin/nvcc" ]]; then
  export CUDA_HOME=${CUDA_HOME:-$_CU13}
  export CUDA_PATH=$CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi
echo "CUDA_HOME=${CUDA_HOME:-unset} nvcc=$(command -v nvcc || echo missing)"
# Reap any dead king leftovers on :8001 / GPU5 (exact PIDs only — never pkill -f)
if [[ -f /root/logs/vllm_king.pid ]]; then
  old=$(cat /root/logs/vllm_king.pid || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
  rm -f /root/logs/vllm_king.pid
fi
for p in $(ss -lptn 'sport = :8001' 2>/dev/null | awk -Fpid= 'NF>1{split($2,a,","); print a[1]}'); do
  kill -9 "$p" 2>/dev/null || true
done
python - <<'PY'
import os, signal, subprocess, time
want = {5}
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
)
idx_to_uuid = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        idx_to_uuid[int(parts[0])] = parts[1]
uuids = {idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    text=True,
)
pids = set()
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2 and parts[0] in uuids:
        try:
            pids.add(int(parts[1]))
        except ValueError:
            pass
print(f"[p4227-king] reap gpu5 pids={sorted(pids)}")
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
time.sleep(2)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
PY
export CUDA_VISIBLE_DEVICES=5
export TRITON_CACHE_DIR=/tmp/triton_king_gpu5_p4227
mkdir -p "$TRITON_CACHE_DIR" /root/logs
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
echo "king launched pid=$(cat /root/logs/vllm_king.pid) CUDA_HOME=$CUDA_HOME"
