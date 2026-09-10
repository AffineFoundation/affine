#!/bin/bash
# King-seat pod bootstrap. Uploaded to /root/king/bootstrap.sh and launched
# under setsid by ops/king-datagen/kingctl.py. Idempotent: safe to re-run.
#
# Serves the current SN120 king for datagen (rollouts `king_*` policies):
# one vLLM 0.28 replica per REPLICAS entry behind an nginx least_conn
# balancer on FRONT_PORT (a Lium-mapped data port, so the datagen pods reach
# it directly over the public IP with the bearer key). Native tool calls
# (qwen3_xml) + reasoning split (qwen3) because three of the harnesses
# (bash / pi / claude_code) drive the model through tool calls, and the
# trace baker expects reasoning in `reasoning_content` like the teacher's.
#
# Contract (via /root/king/env, mode 0600):
#   KING_KEY        bearer every replica enforces (vllm --api-key)
#   SERVED_NAME     model name the endpoint answers to, e.g. king-0ce59769300c
#   DIGEST          sha256 model_digest on https://models.affine.io  (R2 king)
#   HF_MODEL/HF_REV HF repo + revision instead (genesis king), HF_TOKEN
#   FRONT_PORT      nginx listen port (Lium-mapped)
#   REPLICAS        semicolon list of "port:gpus:tp", e.g. "20001:0:1;20002:1:1"
#   MAX_MODEL_LEN, GPU_UTIL, BATCHED_TOKENS, MAX_NUM_SEQS, VLLM_VERSION
#
# Ends in a supervisor loop that relaunches any dead replica.
set -uo pipefail

cd /root
mkdir -p /root/king /root/logs
set -a
source /root/king/env
set +a
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0

log() { echo "[king-boot] $(date -u +%FT%TZ) $*"; }
fail() { echo "$1" > /root/king/bootstrap.failed; log "FATAL $1"; exit 1; }

log "start served=$SERVED_NAME digest=${DIGEST:-} hf=${HF_MODEL:-} replicas=$REPLICAS front=$FRONT_PORT"
rm -f /root/king/ready

# 0. Disk guard (weights ~70 GB).
free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 120 ] && [ ! -d /root/king/model ]; then
  fail "disk ${free_gb}GB < 120GB"
fi

# 1. uv + venv + vllm + prebuilt flashinfer kernels (same stack as the
#    eval pods / teacher swarm; GDN models need the wheels, not a JIT).
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
[ -x /root/venv/bin/python ] || uv venv /root/venv --python 3.12 || fail venv
if ! /root/venv/bin/python -c "import vllm" 2>/dev/null; then
  log "installing vllm==${VLLM_VERSION:-0.28.0}"
  VIRTUAL_ENV=/root/venv uv pip install "vllm==${VLLM_VERSION:-0.28.0}" hf_transfer \
    > /root/logs/pip_vllm.log 2>&1 || { tail -5 /root/logs/pip_vllm.log; fail pip; }
fi
if ! /root/venv/bin/python -c "import flashinfer_jit_cache" 2>/dev/null; then
  log "installing prebuilt flashinfer kernels"
  VIRTUAL_ENV=/root/venv uv pip install "flashinfer-cubin==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl >> /root/logs/pip_vllm.log 2>&1 || fail pip-cubin
  VIRTUAL_ENV=/root/venv uv pip install "flashinfer-jit-cache==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl/cu130 >> /root/logs/pip_vllm.log 2>&1 || fail pip-jitcache
fi
command -v nginx >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq nginx >/dev/null 2>&1 || fail nginx; }

# 2. Weights. R2 king: manifest-driven, sha-verified, resumable, from the
#    PUBLIC bucket (only crowned models live there; no credentials).
#    Genesis king: HF snapshot at the pinned revision.
MODEL_DIR=/root/king/model
if [ -n "${DIGEST:-}" ]; then
  mkdir -p "$MODEL_DIR"
  if [ ! -f "$MODEL_DIR/.complete" ]; then
    log "downloading king $DIGEST from models.affine.io"
    MODEL_DIR="$MODEL_DIR" DIGEST="$DIGEST" /root/venv/bin/python - <<'PY' || fail download
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
print("download complete", flush=True)
PY
    touch "$MODEL_DIR/.complete"
  fi
  SERVE_TARGET="$MODEL_DIR"
  SERVE_REV=()
elif [ -n "${HF_MODEL:-}" ]; then
  log "downloading $HF_MODEL@${HF_REV:-main}"
  /root/venv/bin/python - <<PY || fail download
from huggingface_hub import snapshot_download
import os
snapshot_download("$HF_MODEL", revision="${HF_REV:-main}",
                  token=os.environ.get("HF_TOKEN") or None, max_workers=8)
print("download ok", flush=True)
PY
  SERVE_TARGET="$HF_MODEL"
  SERVE_REV=(--revision "${HF_REV:-main}")
else
  fail "neither DIGEST nor HF_MODEL set"
fi
rm -f /root/king/bootstrap.failed

# 3. nginx front on FRONT_PORT over the replica ports.
IFS=';' read -ra SPECS <<< "$REPLICAS"
UP=""
for spec in "${SPECS[@]}"; do
  IFS=':' read -r port gpus tp <<< "$spec"
  UP="$UP    server 127.0.0.1:$port max_fails=3 fail_timeout=30s;
"
done
cat > /etc/nginx/nginx.conf <<EOF
worker_processes 4;
error_log /root/logs/nginx_error.log warn;
events { worker_connections 4096; }
http {
  upstream king { least_conn;
$UP  }
  # Anthropic-style clients (Claude Code -> vLLM /v1/messages) send the key
  # as x-api-key; vLLM --api-key only reads Authorization: Bearer. Translate.
  map \$http_x_api_key \$king_auth {
    "" \$http_authorization;
    default "Bearer \$http_x_api_key";
  }
  server {
    listen $FRONT_PORT;
    client_max_body_size 200m;
    proxy_read_timeout 3600s; proxy_send_timeout 3600s; proxy_connect_timeout 30s;
    proxy_buffering off; proxy_request_buffering off;
    location / {
      proxy_pass http://king; proxy_http_version 1.1; proxy_set_header Connection "";
      proxy_set_header Authorization \$king_auth;
    }
  }
}
EOF
nginx -t >/dev/null 2>&1 || fail nginx-conf
nginx -s reload 2>/dev/null || nginx

launch_replica() {  # port gpus tp
  local port=$1 gpus=$2 tp=$3
  log "launch replica port=$port gpus=$gpus tp=$tp"
  # Per-replica compile cache: four replicas starting at once race on the
  # shared torch_compile_cache (FileNotFoundError mid-profile, first box
  # 2026-09-10: 2 of 4 died on first launch and came back on relaunch).
  VLLM_CACHE_ROOT="/root/.cache/vllm_$port" \
  CUDA_VISIBLE_DEVICES=$gpus nohup /root/venv/bin/vllm serve "$SERVE_TARGET" "${SERVE_REV[@]}" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --max-num-batched-tokens "${BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${MAX_NUM_SEQS:-32}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --api-key "$KING_KEY" \
    >> "/root/logs/vllm_$port.log" 2>&1 &
  echo $! > "/root/king/replica_$port.pid"
}

replica_up() {
  curl -sf -m 5 -H "Authorization: Bearer $KING_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}
replica_proc_alive() {
  local pidf="/root/king/replica_$1.pid"
  [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null
}

# 4. Supervisor: launch missing replicas, relaunch dead ones, forever.
#    /root/king/ready appears once EVERY replica answered at least once.
declare -A launched_at fails
log "supervising ${#SPECS[@]} replicas"
touch /root/king/bootstrap.done
while true; do
  now=$(date +%s); up=0
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r port gpus tp <<< "$spec"
    if replica_up "$port"; then
      fails[$port]=0; up=$((up+1))
      launched_at[$port]=${launched_at[$port]:-$now}
      continue
    fi
    fails[$port]=$(( ${fails[$port]:-0} + 1 ))
    if replica_proc_alive "$port"; then
      start=${launched_at[$port]:-$now}
      if [ $((now - start)) -gt 1800 ] && [ "${fails[$port]}" -ge 15 ]; then
        log "replica $port wedged — recycling"
        pkill -9 -P "$(cat /root/king/replica_$port.pid)" 2>/dev/null
        kill -9 "$(cat /root/king/replica_$port.pid)" 2>/dev/null
        rm -f "/root/king/replica_$port.pid"; unset "launched_at[$port]"; fails[$port]=0
      fi
      continue
    fi
    launch_replica "$port" "$gpus" "$tp"
    launched_at[$port]=$now; fails[$port]=0
  done
  if [ "$up" = "${#SPECS[@]}" ] && [ ! -f /root/king/ready ]; then
    log "READY $up/${#SPECS[@]} replicas"; touch /root/king/ready
  fi
  sleep 20
done
