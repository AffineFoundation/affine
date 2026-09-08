#!/bin/bash
# Private king-chat pod bootstrap. Runs ON the pod as root from /root/king-chat.
# Idempotent: re-running skips finished steps and (re)starts the supervisors.
#
# What it builds (one 1×H200 pod, nothing public):
#   vLLM 0.28  127.0.0.1:8000  serves the current SN120 king as model
#                              "affine-king", Bearer-key gated (--api-key)
#   Caddy      127.0.0.1:8080  strips the /king URL prefix and forwards to vLLM
#                              (the Cloudflare tunnel on the operator box routes
#                              https://<host>/king/* here over an SSH forward)
#
# Inputs (0600 file written by kingchat.sh over stdin, never on a cmdline):
#   /root/king-chat/.env   KING_API_KEY=...   KING_DIGEST=<sha256 model_digest>
# Weights come from the PUBLIC bucket https://models.affine.io — only a crowned
# model lives there, so no R2 credentials are needed. Every file is checked
# against the size + sha256 in the signed manifest before vLLM sees it.
set -euo pipefail

cd /root/king-chat
# shellcheck disable=SC1091
source /root/king-chat/.env
: "${KING_API_KEY:?}" "${KING_DIGEST:?}"

MODELS_BASE=${MODELS_BASE:-https://models.affine.io/models/sha256}
MODEL_DIR=/root/models/king-${KING_DIGEST:0:12}
VLLM_PORT=8000
CADDY_PORT=8080
export HF_HOME=/root/hf
mkdir -p /root/logs "$MODEL_DIR" "$HF_HOME"
log() { echo "[king-chat] $(date -u +%FT%TZ) $*"; }

# 1. Python env — same stack as affine/evalsrv/bootstrap.sh (vLLM 0.28 +
#    prebuilt flashinfer wheels; GDN models hard-import flashinfer and the bare
#    package would JIT-compile at startup, which needs nvcc the pod lacks).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d /root/venv ]; then
  uv venv /root/venv --python 3.12
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate
if ! python -c "import vllm, importlib.metadata as m; assert m.version('vllm') == '0.28.0'" 2>/dev/null; then
  log "installing vllm 0.28.0"
  uv pip install "vllm==0.28.0" 2>&1 | tee /root/logs/pip_vllm.log | tail -3
  uv pip install "flashinfer-cubin==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl 2>&1 | tail -2
  uv pip install "flashinfer-jit-cache==0.6.16.post3" \
    --index-url https://flashinfer.ai/whl/cu130 2>&1 | tail -2
fi
log "vllm $(python -c 'import importlib.metadata as m; print(m.version("vllm"))') torch $(python -c 'import torch; print(torch.__version__)')"

# 2. Caddy (static binary; no apt repo dance).
if ! command -v caddy >/dev/null 2>&1; then
  log "installing caddy"
  curl -fsSL -o /usr/local/bin/caddy \
    "https://caddyserver.com/api/download?os=linux&arch=amd64"
  chmod +x /usr/local/bin/caddy
fi

# 3. King weights: manifest-driven, verified, resumable.
MANIFEST="$MODEL_DIR/manifest.json"
if [ ! -f "$MODEL_DIR/.complete" ]; then
  log "fetching manifest for $KING_DIGEST"
  curl -fsSL -o "$MANIFEST" "$MODELS_BASE/$KING_DIGEST/manifest.json"
  python - "$MANIFEST" "$KING_DIGEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
if m["model_digest"] != sys.argv[2]:
    raise SystemExit(f"manifest digest {m['model_digest']} != {sys.argv[2]}")
print(f"[king-chat] manifest ok: {m['model_name']} "
      f"{len(m['files'])} files {sum(f['size'] for f in m['files'])/1e9:.1f} GB")
PY
  # Largest shards first, 3 at a time, each resumable (-C -).
  python - "$MANIFEST" <<'PY' > /root/king-chat/.files
import json, sys
m = json.load(open(sys.argv[1]))
for f in sorted(m["files"], key=lambda f: -f["size"]):
    print(f["path"])
PY
  log "downloading $(wc -l < /root/king-chat/.files) files -> $MODEL_DIR"
  export MODELS_BASE KING_DIGEST MODEL_DIR
  xargs -P 3 -I{} bash -c '
    p="{}"; mkdir -p "$(dirname "$MODEL_DIR/$p")"
    for try in 1 2 3 4 5; do
      curl -fsSL -C - --retry 5 --retry-delay 5 -o "$MODEL_DIR/$p" \
        "$MODELS_BASE/$KING_DIGEST/$p" && exit 0
      sleep 10
    done
    echo "[king-chat] FAILED $p" >&2; exit 1' < /root/king-chat/.files
  log "verifying sha256"
  python - "$MANIFEST" "$MODEL_DIR" <<'PY'
import hashlib, json, os, sys
m, root = json.load(open(sys.argv[1])), sys.argv[2]
bad = []
for f in m["files"]:
    p = os.path.join(root, f["path"])
    if not os.path.exists(p) or os.path.getsize(p) != f["size"]:
        bad.append((f["path"], "size")); continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != f["sha256"]:
        bad.append((f["path"], "sha256"))
if bad:
    for p, why in bad:
        os.remove(os.path.join(root, p)) if why == "sha256" and os.path.exists(os.path.join(root, p)) else None
    raise SystemExit(f"[king-chat] integrity failure: {bad} (bad files removed; re-run)")
print(f"[king-chat] all {len(m['files'])} files verified")
PY
  touch "$MODEL_DIR/.complete"
fi

# 4. Caddy: /king/* -> vLLM, everything else 404. Bound to loopback; the only
#    way in is the SSH forward from the operator box.
cat > /root/king-chat/Caddyfile <<EOF
{
	admin off
	auto_https off
}

# ":port" (any Host header — cloudflared forwards Host: <public host>, and a
# host-keyed site block would answer it with an empty 200); bind = loopback.
http://:${CADDY_PORT} {
	bind 127.0.0.1
	handle_path /king/* {
		reverse_proxy 127.0.0.1:${VLLM_PORT} {
			flush_interval -1
			transport http {
				read_timeout 0
				write_timeout 0
			}
		}
	}
	respond 404
}
EOF
caddy validate --config /root/king-chat/Caddyfile --adapter caddyfile

# 5. Supervisors. Kill old ones — loops first so they cannot respawn the
#    children ([b]racket so pkill never matches this script's own cmdline).
pkill -f '[r]un_vllm.sh' || true
pkill -f '[r]un_caddy.sh' || true
pkill -f '[v]llm serve' || true
pkill -f '[c]addy run' || true
sleep 2

# vLLM flags mirror affine/evalsrv/engine.py (chat role) — same kernels the
# eval pods use for this model family — plus a Cursor-friendly context.
# Thinking is OFF by default (2026-09-07): given a system prompt, the king
# never emits </think> — the whole reply, answer included, stays inside the
# think block. A reasoning parser then files it all as `reasoning` and
# Cursor shows an empty message. With enable_thinking=false it answers
# directly; no --reasoning-parser so whatever it emits is visible content.
# Clients can still opt in per request via chat_template_kwargs.
# Tool parser is qwen3_xml, not hermes: the Qwen3.6 template emits
# <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>,
# which hermes (JSON) passes through as plain text.
cat > /root/king-chat/run_vllm.sh <<EOF
#!/bin/bash
# vLLM supervise loop (written by pod_bootstrap.sh)
source /root/venv/bin/activate
source /root/king-chat/.env
export HF_HOME=/root/hf
while true; do
  echo "[king-chat] \$(date -u +%FT%TZ) launching vllm ($MODEL_DIR)"
  vllm serve "$MODEL_DIR" \\
    --host 127.0.0.1 --port ${VLLM_PORT} \\
    --served-model-name affine-king \\
    --api-key "\$KING_API_KEY" \\
    --tensor-parallel-size 1 \\
    --max-model-len 131072 \\
    --gpu-memory-utilization 0.90 \\
    --max-num-batched-tokens 16384 \\
    --attention-backend FLASH_ATTN \\
    --attention-config.use_trtllm_attention 0 \\
    --compilation-config.pass_config.fuse_allreduce_rms false \\
    --moe-backend triton \\
    --additional-config '{"gdn_prefill_backend": "triton"}' \\
    --safetensors-load-strategy prefetch \\
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \\
    --default-chat-template-kwargs '{"enable_thinking": false}' \\
    || echo "[king-chat] vllm exited \$?"
  sleep 10
done
EOF
chmod +x /root/king-chat/run_vllm.sh
cat > /root/king-chat/run_caddy.sh <<'EOF'
#!/bin/bash
# caddy supervise loop (written by pod_bootstrap.sh)
while true; do
  caddy run --config /root/king-chat/Caddyfile --adapter caddyfile \
    || echo "[king-chat] caddy exited $?"
  sleep 5
done
EOF
chmod +x /root/king-chat/run_caddy.sh
nohup /root/king-chat/run_vllm.sh </dev/null >> /root/logs/vllm.log 2>&1 &
nohup /root/king-chat/run_caddy.sh </dev/null >> /root/logs/caddy.log 2>&1 &
log "supervisors started; vllm log /root/logs/vllm.log"
