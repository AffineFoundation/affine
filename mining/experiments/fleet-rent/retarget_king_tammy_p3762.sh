#!/usr/bin/env bash
# p3762: live crown flipped reign34 cryptoDev23 → tammyfritz reign35 @ ~00:26Z.
# Swap :8001 to tammyfritz/Affine-5hmwhnfbix-tammy2@7e5fd5f8…. Teacher untouched.
# Train axes on other GPUs untouched. Never pkill -f.
# Usage on pod: bash retarget_king_tammy_p3762.sh
set -euo pipefail

NEW_REPO=tammyfritz/Affine-5hmwhnfbix-tammy2
NEW_REV=7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
NEW_LOCAL=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/${NEW_REV}
OLD_REPO=cryptoDev23/Affine-5Dku3dYp9j-hk8161
OLD_REV=55b7ffe003d078a8a131673f677b2584548a502e
PREFETCH=${PREFETCH:-1}
SWAP_KING=${SWAP_KING:-1}

mkdir -p /root/logs /root/mining_src/fleet-rent /root/.triton/cache/king
LOG=/root/logs/retarget_tammy_p3762.log
exec >>"$LOG" 2>&1
echo "[p3762] $(date -u +%Y-%m-%dT%H:%M:%SZ) PREFETCH=$PREFETCH SWAP_KING=$SWAP_KING start → $NEW_REPO@$NEW_REV"

set -a
# shellcheck disable=SC1091
source /root/mine.env
set +a
# shellcheck disable=SC1091
source /root/venv/bin/activate
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
# XET finalize can OOM/ENOSPC on full /lium-cipher even mid-snapshot (p3345 crown)
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
_CU13="${_SITE}/nvidia/cu13"
if [[ -x "${_CU13}/bin/nvcc" && -f "${_CU13}/include/cuda_fp16.h" ]]; then
  export CUDA_HOME=${CUDA_HOME:-$_CU13}
  export CUDA_PATH=$CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p3762] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidf=$1
  local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid
  pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"
  rm -f "$pidf"
}

# --- patch mine.env ---
python3 - <<PY
from pathlib import Path
p = Path("/root/mine.env")
text = p.read_text() if p.exists() else ""
lines = text.splitlines()
want = {
    "KING_REPO": "$NEW_REPO",
    "KING_REV": "$NEW_REV",
    "KING_LOCAL": "$NEW_LOCAL",
    "KING_SERVED_NAME": "$NEW_REPO",
    "RESTART_KING": "1",
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
}
seen = set()
out = []
for line in lines:
    if line.startswith("export ") and "=" in line:
        key = line.split("=", 1)[0].removeprefix("export ").strip()
        if key in want:
            out.append(f"export {key}={want[key]}")
            seen.add(key)
            continue
    out.append(line)
for key, val in want.items():
    if key not in seen:
        out.append(f"export {key}={val}")
p.write_text("\n".join(out) + "\n")
print("[p3762] mine.env KING→tammy reign35 RESTART_KING=1 OFFLINE=0")
PY

# --- prefetch ---
hub_ok() {
  # config/index alone is not enough — partial xet fail left 14/16 shards (p3345)
  local n
  n=$(ls "$NEW_LOCAL"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$NEW_LOCAL/config.json" ]] && [[ -f "$NEW_LOCAL/model.safetensors.index.json" ]] && [[ "${n:-0}" -ge 16 ]]
}

if ! hub_ok; then
  if [[ "$PREFETCH" != "1" ]]; then
    echo "[p3762] tammy missing and PREFETCH=0 — abort"
    exit 2
  fi
  echo "[p3762] prefetch $NEW_REPO@$NEW_REV"
  python3 - <<PY
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id="$NEW_REPO",
    revision="$NEW_REV",
    token=os.environ.get("HF_TOKEN"),
)
print("[p3762] snapshot_download ok", path, flush=True)
PY
fi

if [[ ! -d "$NEW_LOCAL" ]]; then
  alt=$(find /root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || true)
  if [[ -n "${alt:-}" ]]; then
    NEW_LOCAL=$alt
    echo "[p3762] NEW_LOCAL fallback $NEW_LOCAL"
  fi
fi
if ! hub_ok && [[ ! -f "$NEW_LOCAL/config.json" ]]; then
  echo "[p3762] FATAL tammy still missing at $NEW_LOCAL"
  exit 2
fi
n=$(ls "$NEW_LOCAL"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
echo "[p3762] tammy shards=$n path=$NEW_LOCAL"

# Tok-style preprocessor landmine
if [[ ! -f "$NEW_LOCAL/preprocessor_config.json" && -f "$NEW_LOCAL/processor_config.json" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("$NEW_LOCAL")
proc = json.loads((p / "processor_config.json").read_text())
out = {
    "processor_class": proc.get("processor_class", "Qwen2VLProcessor"),
    "image_processor_type": "Qwen2VLImageProcessor",
    "auto_map": proc.get("auto_map", {}),
}
(p / "preprocessor_config.json").write_text(json.dumps(out, indent=2) + "\n")
print("[p3762] wrote preprocessor_config.json from processor_config.json")
PY
fi

# Visual key count (informational; multimodal restore is a separate landmine if needed)
if [[ -f "$NEW_LOCAL/model.safetensors.index.json" ]]; then
  python3 - <<PY || true
import json
from pathlib import Path
p = Path("$NEW_LOCAL")
idx = json.loads((p / "model.safetensors.index.json").read_text())
wm = idx.get("weight_map", {})
vis = [k for k in wm if "model.visual" in k or k.startswith("visual.")]
print(f"[p3762] visual keys in index={len(vis)} restored={(p / 'model-visual-restored.safetensors').exists()}")
PY
fi

# --- swap king :8001 GPUs 2,3 ---
if [[ "$SWAP_KING" != "1" ]]; then
  echo "[p3762] SWAP_KING=0 — mine.env patched only"
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/retarget_tammy_p3762.done
  exit 0
fi

# stop current king by pidfile or exact cmdline match to reign34 / :8001
stop_pidfile /root/logs/vllm_king.pid "old king"
for pid in $(pgrep -af 'vllm serve' 2>/dev/null | grep -E "${OLD_REPO}|cryptoDev23|5Dku3dYp9j|hk8161|--port 8001" | awk '{print $1}' || true); do
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -qE 'vllm serve.*(cryptoDev23|5Dku3dYp9j|hk8161|--port 8001)'; then
    stop_pid "$pid" "old reign34 king cmdline"
  fi
done
# also stop Worker_TP on GPUs 2,3 left after API death
for pid in $(pgrep -af 'Worker_TP|VLLM::Worker' 2>/dev/null | awk '{print $1}' || true); do
  envf=/proc/$pid/environ
  [[ -r "$envf" ]] || continue
  cvd=$(tr '\0' '\n' <"$envf" 2>/dev/null | awk -F= '/^CUDA_VISIBLE_DEVICES=/{print $2}')
  if [[ "$cvd" == "2,3" ]]; then
    stop_pid "$pid" "orphan Worker_TP CVD=2,3"
  fi
done
sleep 3

# seed Triton from prior king cache if empty
if [[ ! -d /root/.triton/cache/king ]] || [[ -z "$(ls -A /root/.triton/cache/king 2>/dev/null || true)" ]]; then
  if [[ -d /root/.triton/cache/chall ]] && [[ -n "$(ls -A /root/.triton/cache/chall 2>/dev/null || true)" ]]; then
    cp -a /root/.triton/cache/chall/. /root/.triton/cache/king/ || true
    echo "[p3762] seeded king Triton from chall"
  fi
fi

LOGK=/root/logs/vllm_king.log
PIDF=/root/logs/vllm_king.pid
: >"$LOGK"
echo "[p3762] launch king $NEW_REPO@$NEW_REV GPUs 2,3 :8001"
CUDA_VISIBLE_DEVICES=2,3 TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup vllm serve "$NEW_REPO" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.80 \
    --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --served-model-name "$NEW_REPO" \
    --revision "$NEW_REV" \
    >"$LOGK" 2>&1 &
echo $! >"$PIDF"
echo "[p3762] king pid=$(cat "$PIDF") log=$LOGK"

# wait ready (up to ~20 min)
ready=0
for i in $(seq 1 240); do
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "[p3762] king died — tail log:"
    tail -40 "$LOGK" || true
    exit 3
  fi
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "[p3762] king READY after ${i}0s"
    ready=1
    break
  fi
  if (( i % 12 == 0 )); then
    echo "[p3762] wait king… ${i}0s last=$(tail -1 "$LOGK" 2>/dev/null | cut -c1-120)"
  fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then
  echo "[p3762] FAIL king not ready in 2400s"
  tail -40 "$LOGK" || true
  exit 4
fi

curl -sf http://127.0.0.1:8001/v1/models | head -c 400 || true
echo
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/retarget_tammy_p3762.done
echo "[p3762] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE king=$NEW_REPO@$NEW_REV"
