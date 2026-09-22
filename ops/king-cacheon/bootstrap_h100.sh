#!/bin/bash
# Serve the SN120 king on the Cacheon (SN14) Qwen3.6-35B arena stack — one
# H100 80 GB, SGLang 0.5.19, TP1 — with the crowned `forward_pass` bundle
# applied as soon as its disclosure delay ends. Permissionless: no API key is
# checked (SGLang is launched without --api-key; any bearer, or none, works).
#
# Runs ON the pod as root from /root/cacheon-king. Idempotent.
#
#   PUBLIC_PORT (Lium-mapped)  nginx  ->  127.0.0.1:ENGINE_PORT  SGLang in the
#   arena docker image (lmsysorg/sglang pinned digest + cacheon + H100 MoE
#   tables — examples/arena_inputs/qwen36/Dockerfile, built here), model
#   mounted read-only at /model, bundle at /bundles/champion.
#   Everything the container mounts lives under /srv: on Lium pods /root is a
#   gocryptfs (FUSE) volume that the nested dockerd cannot bind-mount.
#
# Two long-lived loops end the script:
#   run_engine.sh   relaunches the engine; reads /root/cacheon-king/mode
#                   ("stock" | "bundle") and, when present, the extra server
#                   flags in /root/cacheon-king/extra_args (one line; e.g.
#                   speculative decoding) on every launch. Whatever sits in
#                   $DATA/models/draft-mtp is mounted at /draft.
#   bundle_watch.sh polls dash.cacheon.ai from RELEASE_AT, verifies the
#                   bundle's content hash, scans it, flips mode -> bundle and
#                   restarts the engine once.
#
# Inputs (/root/cacheon-king/env, written by the operator side):
#   KING_DIGEST      sha256 model_digest on https://models.affine.io
#   RESERVATION_ID   Cacheon reservation (the winner)
#   CONTENT_HASH     its published bundle content hash (cacheon.bundle_hash)
#   RELEASE_AT       unix time the dashboard discloses the bundle
#   CACHEON_REV      latent-to/cacheon commit to build the arena image from
#   PUBLIC_PORT      Lium-mapped port nginx listens on (default 20000)
set -uo pipefail

ROOT=/root/cacheon-king
DATA=/srv/cacheon-king            # bind-mountable (not the FUSE /root)
cd "$ROOT"
set -a; source "$ROOT/env"; set +a
: "${KING_DIGEST:?}" "${RESERVATION_ID:?}" "${CONTENT_HASH:?}" "${RELEASE_AT:?}" "${CACHEON_REV:?}"
PUBLIC_PORT=${PUBLIC_PORT:-20000}
ENGINE_PORT=30000
IMAGE=cacheon-qwen-h100
BASE_IMAGE="lmsysorg/sglang@sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262"
MODEL_DIR=$DATA/models/king-${KING_DIGEST:0:12}
mkdir -p "$ROOT/logs" "$DATA/bundles" "$DATA/work/receipts" "$MODEL_DIR"
log() { echo "[cacheon-king] $(date -u +%FT%TZ) $*"; }
fail() { echo "$1" > "$ROOT/bootstrap.failed"; log "FATAL $1"; exit 1; }
rm -f "$ROOT/bootstrap.failed"
[ -f "$ROOT/mode" ] || echo stock > "$ROOT/mode"

# 1. Host tools.
export DEBIAN_FRONTEND=noninteractive
command -v nginx >/dev/null || { apt-get update -qq >/dev/null; apt-get install -y -qq nginx git >/dev/null || fail apt; }
command -v git >/dev/null || apt-get install -y -qq git >/dev/null

# 2. Cacheon source at the pinned commit (the .pth bootstrap, seams, CLI).
if [ ! -d "$ROOT/src/.git" ]; then
  git clone -q https://github.com/latent-to/cacheon "$ROOT/src" || fail clone
fi
git -C "$ROOT/src" fetch -q origin && git -C "$ROOT/src" checkout -q "$CACHEON_REV" || fail checkout

# 3. Arena image, exactly as examples/arena_inputs/qwen36/README.md builds it.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "pulling $BASE_IMAGE"
  docker pull -q "$BASE_IMAGE" >> "$ROOT/logs/docker_build.log" 2>&1 || fail pull
  ctx=$(mktemp -d)
  mkdir "$ctx/src"
  git -C "$ROOT/src" archive HEAD | tar -x -C "$ctx/src"
  cp "$ROOT/src/examples/arena_inputs/qwen36/Dockerfile" "$ctx/Dockerfile"
  cp -r "$ROOT/src/examples/arena_inputs/qwen36/moecfg" "$ctx/moecfg"
  log "building $IMAGE"
  docker build --network=none \
    --label "org.opencontainers.image.revision=$CACHEON_REV" \
    -t "$IMAGE" "$ctx" >> "$ROOT/logs/docker_build.log" 2>&1 || { tail -20 "$ROOT/logs/docker_build.log"; fail build; }
  rm -rf "$ctx"
fi
docker run --rm "$IMAGE" python3.12 -I -c 'import cacheon, sglang, torch; print("image ok", sglang.__version__, torch.__version__)' \
  | tee -a "$ROOT/logs/bootstrap.log" || fail image-smoke

# 4. King weights: manifest-driven, sha256-verified, resumable, PUBLIC bucket.
if [ ! -f "$MODEL_DIR/.complete" ]; then
  log "downloading king $KING_DIGEST from models.affine.io"
  MODEL_DIR="$MODEL_DIR" DIGEST="$KING_DIGEST" python3 - <<'PY' || fail download
import hashlib, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
base = f"https://models.affine.io/models/sha256/{os.environ['DIGEST']}/"
man = json.loads(subprocess.run(["curl", "-sSL", "--retry", "5", base + "manifest.json"],
                                check=True, capture_output=True, text=True).stdout)
if man.get("model_digest") not in (None, os.environ["DIGEST"]):
    raise SystemExit("manifest digest mismatch")
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
            os.remove(dst); raise SystemExit(f"sha mismatch {path}")
    return "ok " + path
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(fetch, sorted(man["files"], key=lambda f: -f["size"])):
        print(r, flush=True)
PY
  touch "$MODEL_DIR/.complete"
fi

# 5. nginx: public port -> engine, no auth, request log. Anthropic-style
#    clients send x-api-key; irrelevant here (nothing checks a key).
cat > /etc/nginx/nginx.conf <<EOF
worker_processes 2;
error_log /root/cacheon-king/logs/nginx_error.log warn;
events { worker_connections 2048; }
http {
  log_format king '\$time_iso8601 \$remote_addr "\$request" \$status \$body_bytes_sent \$request_time "\$http_user_agent"';
  access_log /root/cacheon-king/logs/nginx_access.log king;
  server {
    listen $PUBLIC_PORT;
    client_max_body_size 100m;
    proxy_read_timeout 3600s; proxy_send_timeout 3600s; proxy_connect_timeout 30s;
    proxy_buffering off; proxy_request_buffering off;
    location / {
      proxy_pass http://127.0.0.1:$ENGINE_PORT; proxy_http_version 1.1; proxy_set_header Connection "";
    }
  }
}
EOF
nginx -t >/dev/null 2>&1 || fail nginx-conf
nginx -s reload 2>/dev/null || nginx

# 6. Engine supervisor. The arena engine config (engine-config.json) as
#    server flags: FP8 KV, FP32 recurrent state, Triton MoE, CUDA graphs at
#    the arena batch sizes, radix cache off, mem fraction 0.93. Tool-call and
#    reasoning parsers added for chat clients. No --api-key on purpose.
cat > "$ROOT/run_engine.sh" <<EOF
#!/bin/bash
# engine supervise loop (written by bootstrap_h100.sh)
while true; do
  mode=\$(cat $ROOT/mode 2>/dev/null || echo stock)
  extra=()
  if [ "\$mode" = bundle ] && [ -f $DATA/bundles/champion/manifest.toml ]; then
    # PYTHONPATH: the direct loader registers only the entry file; a bundle that
    # imports its own package (from qwen36_layer.moe import ...) needs its root
    # importable — the validator gets this from its materialized engine tree.
    extra=(-e CACHEON_ACTIVE=1 -e CACHEON_BUNDLE_PATH=/bundles/champion -e CACHEON_FRAMEWORK_MODE=0
           -e SGLANG_PLUGINS=cacheon -e CACHEON_SEAM_RECEIPT_DIR=/work/receipts
           -e PYTHONPATH=/bundles/champion)
    rm -f $DATA/work/receipts/*
  fi
  extra_args=\$(cat $ROOT/extra_args 2>/dev/null || true)
  echo "[cacheon-king] \$(date -u +%FT%TZ) launching engine mode=\$mode extra_args=[\$extra_args]"
  docker rm -f king-engine >/dev/null 2>&1
  mkdir -p $DATA/models/draft-mtp
  docker run --rm --name king-engine --gpus all --network host --shm-size 32g --ipc host \\
    -v $MODEL_DIR:/model:ro -v $DATA/models/draft-mtp:/draft:ro -v $DATA/bundles:/bundles:ro -v $DATA/work:/work \\
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "\${extra[@]}" \\
    $IMAGE python3.12 -m sglang.launch_server \\
      --model-path /model --served-model-name affine-king \\
      --host 127.0.0.1 --port $ENGINE_PORT --tp-size 1 \\
      --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 \\
      --max-mamba-cache-size 48 --mem-fraction-static 0.93 \\
      --moe-runner-backend triton --disable-radix-cache \\
      --cuda-graph-bs 1 2 4 8 16 24 32 40 48 \\
      --tool-call-parser qwen3_coder --reasoning-parser qwen3 \\
      --log-level info \$extra_args \\
    || echo "[cacheon-king] engine exited \$?"
  sleep 10
done
EOF
chmod +x "$ROOT/run_engine.sh"

# 7. Bundle watcher: fetch at disclosure, verify content hash, scan, flip.
cat > "$ROOT/bundle_watch.sh" <<EOF
#!/bin/bash
# bundle watcher (written by bootstrap_h100.sh)
URL="https://dash.cacheon.ai/api/submissions/$RESERVATION_ID/bundle.tar.gz?arena=qwen"
log() { echo "[bundle-watch] \$(date -u +%FT%TZ) \$*"; }
while true; do
  if [ -f $ROOT/bundle.applied ]; then log "already applied"; exit 0; fi
  now=\$(date +%s)
  if [ "\$now" -lt "$RELEASE_AT" ]; then sleep \$(( $RELEASE_AT - now < 300 ? $RELEASE_AT - now + 5 : 300 )); continue; fi
  tmp=\$(mktemp -d)
  code=\$(curl -sSL -m 120 -A "Mozilla/5.0 affine-king-cacheon" -o "\$tmp/b.tgz" -w '%{http_code}' "\$URL")
  if [ "\$code" != 200 ]; then log "download HTTP \$code; retry in 120s"; rm -rf "\$tmp"; sleep 120; continue; fi
  tar -xzf "\$tmp/b.tgz" -C "\$tmp" || { log "bad archive"; rm -rf "\$tmp"; sleep 120; continue; }
  dir=\$(find "\$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
  got=\$(PYTHONPATH=$ROOT/src python3 -c "from cacheon.bundle_hash import content_hash; print(content_hash('\$dir'))")
  if [ "\$got" != "$CONTENT_HASH" ]; then log "content hash \$got != $CONTENT_HASH; refusing"; rm -rf "\$tmp"; sleep 600; continue; fi
  rm -rf $DATA/bundles/champion && mv "\$dir" $DATA/bundles/champion && rm -rf "\$tmp"
  log "bundle verified (\$got) -> $DATA/bundles/champion"; ls -la $DATA/bundles/champion; cat $DATA/bundles/champion/manifest.toml
  docker run --rm -v $DATA/bundles:/bundles:ro $IMAGE python3.12 -m cacheon.cli scan /bundles/champion || { log "scan FAILED; staying on stock"; touch $ROOT/bundle.rejected; exit 1; }
  echo bundle > $ROOT/mode
  docker rm -f king-engine >/dev/null 2>&1   # run_engine.sh relaunches with the bundle armed
  touch $ROOT/bundle.applied
  log "mode=bundle; engine restarting"
  exit 0
done
EOF
chmod +x "$ROOT/bundle_watch.sh"

pkill -f '[r]un_engine.sh' || true
pkill -f '[b]undle_watch.sh' || true
docker rm -f king-engine >/dev/null 2>&1
sleep 2
nohup "$ROOT/run_engine.sh" </dev/null >> "$ROOT/logs/engine.log" 2>&1 &
nohup "$ROOT/bundle_watch.sh" </dev/null >> "$ROOT/logs/watch.log" 2>&1 &
touch "$ROOT/bootstrap.done"
log "supervisors started (engine mode=$(cat "$ROOT/mode")); logs in $ROOT/logs"
