#!/usr/bin/env bash
# p4351: live crown flipped reign36 vera6 → 0pentensor reign37 @ ~18:33Z.
# Swap :8001 (GPU2 TP1) to 0pentensor/Affine-5dflhtkufw-awesome-v16@fb6cc85e….
# Teacher untouched. Leave chall/train on other GPUs unless they hold :8001.
# Never pkill -f.
# Usage on pod: bash retarget_king_0pentensor_reign37_p4351.sh
set -euo pipefail

NEW_REPO=0pentensor/Affine-5dflhtkufw-awesome-v16
NEW_REV=fb6cc85e4734119876d9fdaaf9b398b70a15f7ee
NEW_LOCAL=/root/hf/hub/models--0pentensor--Affine-5dflhtkufw-awesome-v16/snapshots/${NEW_REV}
OLD_REPO=vera6/affine-5g4yy75zuz-t6
OLD_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
PREFETCH=${PREFETCH:-1}
SWAP_KING=${SWAP_KING:-1}
KING_GPU=${KING_GPU:-2}
UTIL=${UTIL:-0.85}

mkdir -p /root/logs /root/mining_src/fleet-rent /root/.triton/cache/king
LOG=/root/logs/retarget_0pentensor_reign37_p4351.log
exec >>"$LOG" 2>&1
echo "[p4351] $(date -u +%Y-%m-%dT%H:%M:%SZ) PREFETCH=$PREFETCH SWAP_KING=$SWAP_KING GPU=$KING_GPU UTIL=$UTIL → $NEW_REPO@$NEW_REV"

set -a
# shellcheck disable=SC1091
source /root/mine.env
set +a
# shellcheck disable=SC1091
source /root/venv/bin/activate
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
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
    echo "[p4351] kill pid=$pid ($why)"
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
    elif "=" in line and not line.strip().startswith("#"):
        key = line.split("=", 1)[0].strip()
        if key in want:
            out.append(f"{key}={want[key]}")
            seen.add(key)
            continue
    out.append(line)
for key, val in want.items():
    if key not in seen:
        out.append(f"export {key}={val}")
p.write_text("\n".join(out) + "\n")
print("[p4351] mine.env KING→0pentensor reign37 RESTART_KING=1 OFFLINE=0")
PY

hub_ok() {
  local n want
  n=$(ls "$NEW_LOCAL"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  want=$(python3 -c "import json; print(len(set(json.load(open('$NEW_LOCAL/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 16)
  [[ -f "$NEW_LOCAL/config.json" ]] && [[ -f "$NEW_LOCAL/model.safetensors.index.json" ]] && [[ "${n:-0}" -ge "${want:-16}" ]]
}

if ! hub_ok; then
  if [[ "$PREFETCH" != "1" ]]; then
    echo "[p4351] 0pentensor missing and PREFETCH=0 — abort"
    exit 2
  fi
  echo "[p4351] prefetch $NEW_REPO@$NEW_REV"
  python3 - <<PY
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id="$NEW_REPO",
    revision="$NEW_REV",
    token=os.environ.get("HF_TOKEN"),
)
print("[p4351] snapshot_download ok", path, flush=True)
PY
fi

if [[ ! -d "$NEW_LOCAL" ]]; then
  alt=$(find /root/hf/hub/models--0pentensor--Affine-5dflhtkufw-awesome-v16/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || true)
  if [[ -n "${alt:-}" ]]; then
    NEW_LOCAL=$alt
    echo "[p4351] NEW_LOCAL fallback $NEW_LOCAL"
  fi
fi
if ! hub_ok; then
  echo "[p4351] FATAL 0pentensor still missing at $NEW_LOCAL"
  exit 2
fi
n=$(ls "$NEW_LOCAL"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
echo "[p4351] 0pentensor shards=$n path=$NEW_LOCAL"

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
print("[p4351] wrote preprocessor_config.json from processor_config.json")
PY
fi

if [[ "$SWAP_KING" != "1" ]]; then
  echo "[p4351] SWAP_KING=0 — mine.env patched + prefetch only"
  date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/retarget_0pentensor_reign37_p4351.prefetch.done
  exit 0
fi

stop_pidfile /root/logs/vllm_king.pid "old king"
# Exact cmdline kill for :8001 / old vera king — never pkill -f
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -qE 'vllm serve.*(vera6|5g4yy75zuz|--port 8001)'; then
    stop_pid "$pid" "old reign36 king cmdline"
  fi
done < <(ps -eo pid=,args= | awk '/vllm serve / && !/awk/ {print $1}')

# Orphan Worker_TP on KING_GPU only
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  envf=/proc/$pid/environ
  [[ -r "$envf" ]] || continue
  cvd=$(tr '\0' '\n' <"$envf" 2>/dev/null | awk -F= '/^CUDA_VISIBLE_DEVICES=/{print $2}')
  if [[ "$cvd" == "$KING_GPU" ]]; then
    cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -qE 'Worker_TP|VLLM::Worker|vllm'; then
      stop_pid "$pid" "orphan worker CVD=$KING_GPU"
    fi
  fi
done < <(ps -eo pid=,args= | awk '/Worker_TP|VLLM::Worker/ && !/awk/ {print $1}')
sleep 3

for i in $(seq 1 60); do
  used=$(nvidia-smi -i "$KING_GPU" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    echo "[p4351] GPU$KING_GPU free used_mib=$used"
    break
  fi
  sleep 2
done

if [[ ! -d /root/.triton/cache/king ]] || [[ -z "$(ls -A /root/.triton/cache/king 2>/dev/null || true)" ]]; then
  if [[ -d /root/.triton/cache/chall ]] && [[ -n "$(ls -A /root/.triton/cache/chall 2>/dev/null || true)" ]]; then
    cp -a /root/.triton/cache/chall/. /root/.triton/cache/king/ || true
    echo "[p4351] seeded king Triton from chall"
  fi
fi

LOGK=/root/logs/vllm_king.log
PIDF=/root/logs/vllm_king.pid
: >"$LOGK"
echo "[p4351] launch king $NEW_REPO@$NEW_REV GPU $KING_GPU :8001 util=$UTIL"
CUDA_VISIBLE_DEVICES=$KING_GPU TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup vllm serve "$NEW_LOCAL" \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --gpu-memory-utilization "$UTIL" \
    --max-num-batched-tokens 8192 \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    --enforce-eager \
    --served-model-name "$NEW_REPO" \
    >"$LOGK" 2>&1 &
echo $! >"$PIDF"
echo "[p4351] king pid=$(cat "$PIDF") log=$LOGK"

ready=0
for i in $(seq 1 360); do
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "[p4351] king died — tail log:"
    tail -40 "$LOGK" || true
    exit 3
  fi
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
    echo "[p4351] king READY after ${i}0s id=$kid"
    ready=1
    break
  fi
  if (( i % 12 == 0 )); then
    echo "[p4351] wait king… ${i}0s last=$(tail -1 "$LOGK" 2>/dev/null | cut -c1-120)"
  fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then
  echo "[p4351] FAIL king not ready in 3600s"
  tail -40 "$LOGK" || true
  exit 4
fi

curl -sf http://127.0.0.1:8001/v1/models | head -c 400 || true
echo
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/retarget_0pentensor_reign37_p4351.done
echo "[p4351] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE king=$NEW_REPO@$NEW_REV"
