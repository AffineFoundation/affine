#!/bin/bash
# Teacher-swarm pod bootstrap. Uploaded to /root/swarm/bootstrap.sh and run
# under nohup by the manager. Idempotent: safe to re-run.
#
# Contract (via /root/swarm/env, mode 0600):
#   HF_TOKEN        HuggingFace token for the model pull
#   SWARM_KEY       api key every replica enforces (vllm --api-key)
#   MODEL           e.g. zai-org/GLM-4.5-Air-FP8
#   VLLM_VERSION    e.g. 0.22.1
#   REPLICAS        semicolon list of "port:gpus:tp", e.g. "20000:0,1:2;20001:2,3:2"
#   MAX_MODEL_LEN, GPU_UTIL, BATCHED_TOKENS
#
# Ends in a supervisor loop that relaunches any dead replica (pod-local
# self-healing; pod-level healing is the manager's job).
set -uo pipefail

cd /root
mkdir -p /root/swarm /root/logs
set -a  # auto-export: python children need HF_TOKEN etc.
source /root/swarm/env
set +a
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
# Production _vllm_env: keep FlashInfer JIT paths off everywhere.
# DeepGEMM: vLLM 0.22.1 on SM103 (B300) hard-fails FP8 without the optional
# deep_gemm package; force the Triton FP8 fallback instead.
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0

log() { echo "[swarm-boot] $(date -u +%FT%TZ) $*"; }

log "start model=$MODEL vllm=$VLLM_VERSION replicas=$REPLICAS"

# 0. Disk guard: weights are ~110 GB; refuse quietly-broken pods early.
free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 150 ] && [ ! -d "$HF_HOME/hub" ]; then
  log "FATAL disk ${free_gb}GB < 150GB"
  echo "disk" > /root/swarm/bootstrap.failed
  exit 1
fi

# 1. uv + venv + vllm (pinned to production version).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x /root/swarm-venv/bin/python ]; then
  uv venv /root/swarm-venv --python 3.12 || { echo venv > /root/swarm/bootstrap.failed; exit 1; }
fi
if ! /root/swarm-venv/bin/python -c "import vllm" 2>/dev/null; then
  log "installing vllm==$VLLM_VERSION"
  VIRTUAL_ENV=/root/swarm-venv uv pip install "vllm==$VLLM_VERSION" hf_transfer \
    > /root/logs/pip_vllm.log 2>&1 || { tail -5 /root/logs/pip_vllm.log; echo pip > /root/swarm/bootstrap.failed; exit 1; }
fi
# Prebuilt flashinfer kernels (2026-08-27, Qwen3.8 teacher): vLLM >= 0.28
# imports flashinfer for GDN models; without these wheels it JIT-compiles at
# startup and dies on toolkit-less pods ("Could not find nvcc"). Version pins
# match vLLM 0.28.0's flashinfer-python==0.6.16.post3; cu130 matches its torch.
if ! /root/swarm-venv/bin/python -c "import flashinfer_jit_cache" 2>/dev/null; then
  log "installing prebuilt flashinfer kernels"
  VIRTUAL_ENV=/root/swarm-venv uv pip install \
    "flashinfer-cubin==0.6.16.post3" --index-url https://flashinfer.ai/whl \
    >> /root/logs/pip_vllm.log 2>&1 || { echo pip-cubin > /root/swarm/bootstrap.failed; exit 1; }
  VIRTUAL_ENV=/root/swarm-venv uv pip install \
    "flashinfer-jit-cache==0.6.16.post3" --index-url https://flashinfer.ai/whl/cu130 \
    >> /root/logs/pip_vllm.log 2>&1 || { echo pip-jitcache > /root/swarm/bootstrap.failed; exit 1; }
fi

# 2. Model snapshot (resumable; skips instantly when already complete).
log "downloading $MODEL"
/root/swarm-venv/bin/python - <<PY || { echo download > /root/swarm/bootstrap.failed; exit 1; }
from huggingface_hub import snapshot_download
import os
snapshot_download("$MODEL", token=os.environ.get("HF_TOKEN") or None,
                  max_workers=8)
print("download ok", flush=True)
PY

rm -f /root/swarm/bootstrap.failed

launch_replica() {  # port gpus tp
  local port=$1 gpus=$2 tp=$3
  log "launch replica port=$port gpus=$gpus tp=$tp"
  CUDA_VISIBLE_DEVICES=$gpus nohup /root/swarm-venv/bin/vllm serve "$MODEL" \
    --port "$port" \
    --tensor-parallel-size "$tp" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-batched-tokens "$BATCHED_TOKENS" \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --api-key "$SWARM_KEY" \
    >> "/root/logs/vllm_$port.log" 2>&1 &
  echo $! > "/root/swarm/replica_$port.pid"
}

replica_up() {  # port -> 0 if HTTP-ready
  curl -sf -m 5 -H "Authorization: Bearer $SWARM_KEY" \
    "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

replica_proc_alive() {  # port -> 0 if the launched pid is still running
  local pidf="/root/swarm/replica_$1.pid"
  [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null
}

# 3. Supervisor: launch missing replicas, relaunch dead ones, forever.
#    Recycle a live-but-unready process only after it has BOTH been up for
#    30 min since launch AND failed 15 consecutive probes (~5 min) — a busy
#    replica that is slow on /v1/models must not get shot mid-bench.
declare -A launched_at fails
IFS=';' read -ra SPECS <<< "$REPLICAS"
log "supervising ${#SPECS[@]} replicas"
touch /root/swarm/bootstrap.done
while true; do
  now=$(date +%s)
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r port gpus tp <<< "$spec"
    if replica_up "$port"; then
      fails[$port]=0
      launched_at[$port]=${launched_at[$port]:-$now}
      continue
    fi
    fails[$port]=$(( ${fails[$port]:-0} + 1 ))
    if replica_proc_alive "$port"; then
      start=${launched_at[$port]:-$now}
      if [ $((now - start)) -gt 1800 ] && [ "${fails[$port]}" -ge 15 ]; then
        log "replica $port wedged (alive, ${fails[$port]} failed probes) — recycling"
        pkill -9 -P "$(cat /root/swarm/replica_$port.pid)" 2>/dev/null
        kill -9 "$(cat /root/swarm/replica_$port.pid)" 2>/dev/null
        rm -f "/root/swarm/replica_$port.pid"
        unset "launched_at[$port]"
        fails[$port]=0
      fi
      continue
    fi
    launch_replica "$port" "$gpus" "$tp"
    launched_at[$port]=$now
    fails[$port]=0
  done
  sleep 20
done
