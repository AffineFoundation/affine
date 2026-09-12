#!/bin/bash
# Benchmark-suite serving bootstrap for one rented GPU pod (Prime Intellect
# pod or Lium box). Serves TWO models side by side with the chat-box stack
# — vLLM 0.28.0, qwen3 reasoning parser, qwen3_xml tool parser — behind one
# nginx each on loopback:
#
#   king    (Affine king, sha256 digest on https://models.affine.io)  -> 127.0.0.1:8001
#   teacher (HF repo, e.g. Qwen/Qwen3.8-27B)                          -> 127.0.0.1:8002
#
# Contract via /root/bench/env (mode 0600):
#   KING_DIGEST        sha256 model_digest of the king (public bucket, no creds)
#   TEACHER_HF         HF repo id of the teacher; TEACHER_REV optional; HF_TOKEN optional
#   KING_REPLICAS      "port:gpus:tp;..." e.g. "31001:0,1:2;31002:2,3:2"
#   TEACHER_REPLICAS   e.g. "32001:4,5:2;32002:6,7:2"   (empty = do not serve the teacher)
#   API_KEY            bearer every replica enforces
#   MAX_MODEL_LEN, GPU_UTIL, BATCHED_TOKENS, MAX_NUM_SEQS, VLLM_VERSION
#
# Idempotent. Ends in a supervisor loop; /root/bench/ready appears once every
# replica of both models answered /v1/models.
set -uo pipefail
cd /root
mkdir -p /root/bench /root/logs
set -a
source /root/bench/env
set +a
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0

log() { echo "[bench-boot] $(date -u +%FT%TZ) $*"; }
fail() { echo "$1" > /root/bench/bootstrap.failed; log "FATAL $1"; exit 1; }
rm -f /root/bench/ready /root/bench/bootstrap.failed

# 1. uv + venv + vllm + prebuilt flashinfer kernels (same stack as the eval
#    pods / teacher swarm / king seat).
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
[ -x /root/venv/bin/python ] || uv venv /root/venv --python 3.12 || fail venv
# VLLM_CUDA selects the wheel flavour: "cu130" (PyPI default, needs driver
# >= 580) or "cu129" (GitHub release wheel + the cu129 torch index; works on
# the 570-series drivers of the Prime/Lambda A100 hosts via CUDA minor-version
# compatibility).
VLLM_CUDA=${VLLM_CUDA:-cu130}
VV=${VLLM_VERSION:-0.28.0}
if ! /root/venv/bin/python -c "import vllm" 2>/dev/null; then
  log "installing vllm==$VV ($VLLM_CUDA)"
  if [ "$VLLM_CUDA" = "cu129" ]; then
    VIRTUAL_ENV=/root/venv uv pip install \
      "vllm @ https://github.com/vllm-project/vllm/releases/download/v$VV/vllm-$VV+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      hf_transfer huggingface_hub \
      --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      > /root/logs/pip_vllm.log 2>&1 || { tail -5 /root/logs/pip_vllm.log; fail pip; }
  else
    VIRTUAL_ENV=/root/venv uv pip install "vllm==$VV" hf_transfer huggingface_hub \
      > /root/logs/pip_vllm.log 2>&1 || { tail -5 /root/logs/pip_vllm.log; fail pip; }
  fi
fi
if ! /root/venv/bin/python -c "import flashinfer_jit_cache" 2>/dev/null; then
  log "installing prebuilt flashinfer kernels ($VLLM_CUDA)"
  VIRTUAL_ENV=/root/venv uv pip install "flashinfer-cubin==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl >> /root/logs/pip_vllm.log 2>&1 || fail pip-cubin
  VIRTUAL_ENV=/root/venv uv pip install "flashinfer-jit-cache==0.6.16.post3" \
    --index-url "https://flashinfer.ai/whl/$VLLM_CUDA" >> /root/logs/pip_vllm.log 2>&1 || fail pip-jitcache
fi
command -v nginx >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq nginx >/dev/null 2>&1 || fail nginx; }

# 2. King weights: manifest-driven, sha-verified, resumable, PUBLIC bucket.
KING_DIR=/root/bench/king
if [ ! -f "$KING_DIR/.complete" ]; then
  mkdir -p "$KING_DIR"
  log "downloading king $KING_DIGEST from models.affine.io"
  MODEL_DIR="$KING_DIR" DIGEST="$KING_DIGEST" /root/venv/bin/python - <<'PY' || fail download-king
import hashlib, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
base = f"https://models.affine.io/models/sha256/{os.environ['DIGEST']}/"
man = json.loads(subprocess.run(["curl", "-sSL", "--retry", "5", base + "manifest.json"],
                                check=True, capture_output=True, text=True).stdout)
if man.get("model_digest") not in (None, os.environ["DIGEST"]):
    raise SystemExit(f"manifest digest {man.get('model_digest')} != {os.environ['DIGEST']}")
d = os.environ["MODEL_DIR"]

def fetch(f):
    path, size, sha = f["path"], f["size"], f.get("sha256")
    dst = os.path.join(d, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "cached " + path
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["curl", "-sSL", "--http1.1", "--retry", "8", "--retry-all-errors",
                    "-C", "-", "-o", dst, base + path], check=True)
    if sha:
        h = hashlib.sha256()
        with open(dst, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            os.remove(dst)
            raise SystemExit(f"sha mismatch {path}")
    return "ok " + path

with ThreadPoolExecutor(8) as ex:
    for r in ex.map(fetch, sorted(man["files"], key=lambda f: -f["size"])):
        print(r, flush=True)
json.dump(man, open(os.path.join(d, ".manifest.json"), "w"))
print("download complete", flush=True)
PY
  touch "$KING_DIR/.complete"
fi

# 3. Teacher weights: HF snapshot at the pinned revision.
TEACHER_DIR=""
if [ -n "${TEACHER_REPLICAS:-}" ]; then
  log "downloading teacher $TEACHER_HF@${TEACHER_REV:-main}"
  cat > /root/bench/dl_teacher.py <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ["TEACHER_HF"], revision=os.environ.get("TEACHER_REV") or "main",
                      token=os.environ.get("HF_TOKEN") or None, max_workers=8,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
print(p)
PY
  /root/venv/bin/python /root/bench/dl_teacher.py > /root/bench/teacher_dir.txt 2>> /root/logs/dl_teacher.log || fail download-teacher
  TEACHER_DIR=$(tail -1 /root/bench/teacher_dir.txt)
  log "teacher snapshot at $TEACHER_DIR"
fi

# 4. nginx: one least_conn upstream per model on loopback.
upstream_block() {  # name replicas
  local name=$1 specs=$2 up=""
  IFS=';' read -ra arr <<< "$specs"
  for spec in "${arr[@]}"; do
    IFS=':' read -r port gpus tp <<< "$spec"
    up="$up    server 127.0.0.1:$port max_fails=3 fail_timeout=30s;
"
  done
  echo "  upstream $name { least_conn;
$up  }"
}
server_block() {  # listen upstream
  echo "  server {
    listen 127.0.0.1:$1;
    client_max_body_size 200m;
    proxy_read_timeout 3600s; proxy_send_timeout 3600s; proxy_connect_timeout 30s;
    proxy_buffering off; proxy_request_buffering off;
    location / { proxy_pass http://$2; proxy_http_version 1.1; proxy_set_header Connection \"\"; }
  }"
}
{
  echo "worker_processes 4;
error_log /root/logs/nginx_error.log warn;
events { worker_connections 4096; }
http {"
  upstream_block king "$KING_REPLICAS"
  server_block 8001 king
  if [ -n "${TEACHER_REPLICAS:-}" ]; then
    upstream_block teacher "$TEACHER_REPLICAS"
    server_block 8002 teacher
  fi
  echo "}"
} > /etc/nginx/nginx.conf
nginx -t >/dev/null 2>&1 || fail nginx-conf
nginx -s reload 2>/dev/null || nginx

launch_replica() {  # served_name model_path port gpus tp
  local served=$1 model=$2 port=$3 gpus=$4 tp=$5
  log "launch $served replica port=$port gpus=$gpus tp=$tp"
  VLLM_CACHE_ROOT="/root/.cache/vllm_$port" \
  CUDA_VISIBLE_DEVICES=$gpus nohup /root/venv/bin/vllm serve "$model" \
    --served-model-name "$served" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --max-num-batched-tokens "${BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --api-key "$API_KEY" \
    >> "/root/logs/vllm_$port.log" 2>&1 &
  echo $! > "/root/bench/replica_$port.pid"
}
replica_up() {
  curl -sf -m 5 -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}
replica_proc_alive() {
  local pidf="/root/bench/replica_$1.pid"
  [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null
}

# 5. Supervisor over both models' replicas.
declare -a ALL
IFS=';' read -ra K <<< "$KING_REPLICAS"
for s in "${K[@]}"; do ALL+=("king:$KING_DIR:$s"); done
if [ -n "${TEACHER_REPLICAS:-}" ]; then
  IFS=';' read -ra T <<< "$TEACHER_REPLICAS"
  for s in "${T[@]}"; do ALL+=("teacher:$TEACHER_DIR:$s"); done
fi
declare -A launched_at fails
log "supervising ${#ALL[@]} replicas"
touch /root/bench/bootstrap.done
while true; do
  now=$(date +%s); up=0
  for entry in "${ALL[@]}"; do
    IFS=':' read -r served model port gpus tp <<< "$entry"
    if replica_up "$port"; then
      fails[$port]=0; up=$((up+1)); launched_at[$port]=${launched_at[$port]:-$now}; continue
    fi
    fails[$port]=$(( ${fails[$port]:-0} + 1 ))
    if replica_proc_alive "$port"; then
      start=${launched_at[$port]:-$now}
      if [ $((now - start)) -gt 1800 ] && [ "${fails[$port]}" -ge 15 ]; then
        log "replica $port wedged — recycling"
        pkill -9 -P "$(cat /root/bench/replica_$port.pid)" 2>/dev/null
        kill -9 "$(cat /root/bench/replica_$port.pid)" 2>/dev/null
        rm -f "/root/bench/replica_$port.pid"; unset "launched_at[$port]"; fails[$port]=0
      fi
      continue
    fi
    launch_replica "$served" "$model" "$port" "$gpus" "$tp"
    launched_at[$port]=$now; fails[$port]=0
  done
  if [ "$up" = "${#ALL[@]}" ] && [ ! -f /root/bench/ready ]; then
    log "READY $up/${#ALL[@]} replicas"; touch /root/bench/ready
  fi
  sleep 20
done
