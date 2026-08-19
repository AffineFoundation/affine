#!/usr/bin/env bash
# p4032: mine-crown-1 cold TK on idle GPUs 0–3 + arm R912/R913 MERGE→n80 waiters.
# Do not touch R912/R913 TRAIN on GPUs 4–7. Never pkill -f.
# Teacher :8000 GPU0 TP=1 · king reign36 vera :8001 GPU2 TP=1 (B300; matches R888 working pattern).
set -euo pipefail
LOG=/root/logs/p4032_crown_cold_tk.log
mkdir -p /root/logs /root/affine_data /root/.triton/cache/teacher /root/.triton/cache/king
exec > >(tee -a "$LOG") 2>&1
echo "[p4032-crown-tk] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_TOKEN
[[ -n "${HF_TOKEN:-}" ]] || { echo FATAL missing HF_TOKEN; exit 1; }

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

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/${KING_REV}
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_PIDF=/root/logs/vllm_teacher.pid
KING_PIDF=/root/logs/vllm_king.pid
TEACHER_LOG=/root/logs/vllm_teacher.log
KING_LOG=/root/logs/vllm_king.log

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4032-crown-tk] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why pidf=$pidf"
  rm -f "$pidf"
}

hub_ok() {
  local path=$1 min=${2:-1} n
  n=$(ls "$path"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$path/config.json" ]] && [[ "${n:-0}" -ge "$min" ]]
}

wait_ready() {
  local port=$1 name=$2 pidf=$3 logf=$4 ready=0
  for i in $(seq 1 720); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1; echo "[p4032-crown-tk] ${name}_READY poll=$i"; break
    fi
    if [[ -f "$pidf" ]]; then
      local pid; pid=$(cat "$pidf")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[p4032-crown-tk] ERROR $name died"; tail -80 "$logf"; exit 1
      fi
    fi
    (( i % 12 == 0 )) && echo "[p4032-crown-tk] wait $name :$port iter=$i last=$(tail -1 "$logf" 2>/dev/null | cut -c1-120)"
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo "[p4032-crown-tk] ERROR $name not ready"; tail -80 "$logf"; exit 1; }
}

# Refuse if train GPUs 4–7 are somehow free of our trains AND someone else holds 0–3.
for idx in 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print $1+0}')
  echo "[p4032-crown-tk] gpu$idx used_mib=$used"
done
for idx in 0 1 2 3; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print $1+0}')
  if [[ "$used" -ge 8192 ]]; then
    echo "[p4032-crown-tk] FATAL GPU $idx busy used_mib=$used — abort TK"
    exit 2
  fi
done

hub_ok "$KING_LOCAL" 2 || { echo FATAL king incomplete at $KING_LOCAL; exit 3; }
echo "[p4032-crown-tk] king shards=$(ls "$KING_LOCAL"/model-*-of-*.safetensors | wc -l)"

echo "[p4032-crown-tk] DOWNLOAD teacher start $TEACHER_REPO"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
token=os.environ["HF_TOKEN"]
path=snapshot_download("zai-org/GLM-4.5-Air-FP8", token=token)
print("[p4032-crown-tk] DOWNLOAD teacher done", path, flush=True)
open("/root/logs/teacher_dl.done","w").write(path+"\n")
PY

# Pin mine.env king → reign36 vera for later n80 waiters.
python3 - <<PY
from pathlib import Path
p = Path("/root/mine.env")
text = p.read_text() if p.exists() else ""
lines = text.splitlines()
want = {
    "KING_REPO": "$KING_REPO",
    "KING_REV": "$KING_REV",
    "KING_LOCAL": "$KING_LOCAL",
    "KING_SERVED_NAME": "$KING_REPO",
    "TEACHER_REPO": "$TEACHER_REPO",
    "SKIP_LOCAL_TKC": "0",
}
seen=set(); out=[]
for line in lines:
    if line.startswith("export ") and "=" in line:
        key=line.split("=",1)[0].removeprefix("export ").strip()
        if key in want:
            out.append(f"export {key}={want[key]}"); seen.add(key); continue
    # also rewrite bare KEY=VAL (mine.env style without export)
    if "=" in line and not line.strip().startswith("#"):
        key=line.split("=",1)[0].strip()
        if key in want and not key.startswith("export "):
            out.append(f"{key}={want[key]}"); seen.add(key); continue
    out.append(line)
for key,val in want.items():
    if key not in seen:
        out.append(f"{key}={val}")
p.write_text("\n".join(out)+"\n")
print("[p4032-crown-tk] mine.env KING→vera reign36")
PY

stop_pidfile "$TEACHER_PIDF" "stale teacher"
stop_pidfile "$KING_PIDF" "stale king"
for port in 8000 8001; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    stop_pid "$pid" "listener :$port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)
done
sleep 2

TEACHER_COMMON=(--tensor-parallel-size 1 --max-model-len 65536 --max-num-batched-tokens 8192
  --attention-backend FLASH_ATTN --attention-config.use_trtllm_attention 0
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton
  --additional-config '{"gdn_prefill_backend": "triton"}'
  --enforce-eager)
KING_COMMON=("${TEACHER_COMMON[@]}")

echo "[p4032-crown-tk] launch teacher :8000 GPU0 TP=1"
: >"$TEACHER_LOG"
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/root/.triton/cache/teacher \
  nohup /root/venv/bin/vllm serve "$TEACHER_REPO" \
  --port 8000 --gpu-memory-utilization 0.90 \
  "${TEACHER_COMMON[@]}" --served-model-name "$TEACHER_REPO" >>"$TEACHER_LOG" 2>&1 &
echo $! >"$TEACHER_PIDF"
echo "[p4032-crown-tk] teacher pid=$(cat "$TEACHER_PIDF")"
wait_ready 8000 teacher "$TEACHER_PIDF" "$TEACHER_LOG"

echo "[p4032-crown-tk] launch king :8001 GPU2 TP=1 $KING_REPO@$KING_REV"
: >"$KING_LOG"
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/root/.triton/cache/king \
  nohup /root/venv/bin/vllm serve "$KING_LOCAL" \
  --port 8001 --gpu-memory-utilization 0.90 \
  "${KING_COMMON[@]}" --served-model-name "$KING_REPO" >>"$KING_LOG" 2>&1 &
echo $! >"$KING_PIDF"
echo "[p4032-crown-tk] king pid=$(cat "$KING_PIDF")"
wait_ready 8001 king "$KING_PIDF" "$KING_LOG"

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "[p4032-crown-tk] king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { echo ERROR king not vera; exit 5; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/crown_tk_ready_p4032.done

# Arm MERGE→n80 waiters (train→merge already armed; n80 was missing).
for pair in \
  "r912:/root/mining_src/r912-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/wait_r912_merge_then_n80_p4017.sh:/root/logs/p4032_r912_n80_wait.outer.pid" \
  "r913:/root/mining_src/r913-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr/wait_r913_merge_then_n80_p4017.sh:/root/logs/p4032_r913_n80_wait.outer.pid"
do
  IFS=: read -r tag script pidf <<<"$pair"
  if [[ ! -f "$script" ]]; then
    echo "[p4032-crown-tk] WARN missing $script"
    continue
  fi
  chmod +x "$script"
  if [[ -f "$pidf" ]]; then
    old=$(cat "$pidf" 2>/dev/null || true)
    if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
      echo "[p4032-crown-tk] $tag n80 waiter already alive pid=$old"
      continue
    fi
  fi
  nohup bash "$script" >/root/logs/p4032_${tag}_n80_wait.outer.nohup 2>&1 &
  echo $! >"$pidf"
  echo "[p4032-crown-tk] armed $tag MERGE→n80 pid=$(cat "$pidf")"
done

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4032_crown_cold_tk_armed.done
echo "[p4032-crown-tk] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
